#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>   // For automatic STL container bindings
#include <pybind11/numpy.h> // For NumPy array support (if needed)
#include "ompl/datastructures/NearestNeighborsGNAT.h"

#include <vector>
#include <memory>
#include <cmath>
#include <iostream>
#include <fstream>
#include <functional>

namespace py = pybind11;

// Define a serializable vector type that will be used with the GNAT.
template <typename T>
struct vectorSerializable : public std::vector<T>
{
    using std::vector<T>::vector;
    vectorSerializable(const std::vector<T>& vec) : std::vector<T>(vec) {}

    // Overload operator<< for serialization
    friend std::ostream& operator<<(std::ostream& os, const vectorSerializable& vec)
    {
        for (const auto& elem : vec)
            os << elem << " ";
        return os;
    }
    // Overload operator>> for deserialization
    friend std::istream& operator>>(std::istream& is, vectorSerializable& vec)
    {
        vec.clear();
        T value;
        while (is >> value)
            vec.push_back(value);
        return is;
    }
};

// GNAT is more convenient to be used with shared pointers
template <typename T>
using vectorSerializablePtr = std::shared_ptr<vectorSerializable<T>>;

// Create an alias for the NearestNeighborsGNAT instantiated with Point
using point = vectorSerializable<double>;
using pointPtr = vectorSerializablePtr<double>;
using GNATVector = ompl::NearestNeighborsGNAT<pointPtr>;

// Helper function to convert a single Python object to PointPtr
std::shared_ptr<point> convert_to_point_ptr(py::object obj) {
    if (py::isinstance<py::array>(obj)) {
        // Handle NumPy array
        py::array_t<double, py::array::c_style | py::array::forcecast> arr = obj.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        if (arr.ndim() != 1) {
            throw std::invalid_argument("Data point NumPy array must be 1D.");
        }
        std::vector<double> vec(arr.size());
        std::memcpy(vec.data(), arr.data(), arr.size() * sizeof(double));
        return std::make_shared<point>(vec);
    }
    else if (py::isinstance<py::list>(obj)) {
        // Handle Python list
        py::list lst = obj.cast<py::list>();
        std::vector<double> vec;
        vec.reserve(lst.size());
        for(auto elem : lst) {
            vec.push_back(elem.cast<double>());
        }
        return std::make_shared<point>(vec);
    }
    else {
        throw std::invalid_argument("Data point must be a 1D NumPy array or a list of floats.");
    }
}

// Helper function to convert a Python object (list of points or 2D NumPy array) to vector<PointPtr>
std::vector<std::shared_ptr<point>> convert_to_point_ptr_list(py::object data) {
    std::vector<std::shared_ptr<point>> data_vec;

    if (py::isinstance<py::array>(data)) {
        // Handle 2D NumPy array
        py::array_t<double, py::array::c_style | py::array::forcecast> arr = data.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        if (arr.ndim() != 2) {
            throw std::invalid_argument("add_list expects a 2D NumPy array or a list of 1D arrays/lists.");
        }

        auto num_rows = arr.shape(0);
        auto num_cols = arr.shape(1);

        data_vec.reserve(num_rows);
        for (auto i = 0; i < num_rows; ++i) {
            std::vector<double> vec(num_cols);
            std::memcpy(vec.data(), arr.data(i, 0), num_cols * sizeof(double));
            data_vec.emplace_back(std::make_shared<point>(vec));
        }
    }
    else if (py::isinstance<py::list>(data)) {
        // Handle Python list of points
        py::list py_list = data.cast<py::list>();
        data_vec.reserve(py_list.size());
        for(auto item : py_list)
        {
            // 'item' is a py::handle, need to cast to py::object
            py::object obj = py::reinterpret_borrow<py::object>(item);
            data_vec.emplace_back(convert_to_point_ptr(obj));
        }
    }
    else {
        throw std::invalid_argument("add_list expects a 2D NumPy array or a list of 1D arrays/lists.");
    }

    return data_vec;
}

PYBIND11_MODULE(_gnat, m)
{
    // Expose the VectorSerializable class to Python
    py::class_<point, std::shared_ptr<point>>(m, "Vector")
        .def(py::init<>())
        .def("__repr__", [](const point& vec) {
            std::ostringstream oss;
            oss << vec;
            return oss.str();
        })
        .def("__getitem__", [](const point &v, size_t i) -> double {
            if (i >= v.size())
                throw py::index_error();
            return v[i];
        })
        .def("__len__", &point::size)
        .def("__iter__", [](const point &v) {
            return py::make_iterator(v.begin(), v.end());
        }, py::keep_alive<0, 1>()) // Keep vector alive while iterator is used
        .def("push_back", [](point &self, double value) {
            self.push_back(value);
        })
        .def("clear", &point::clear);

    // Expose the GNATVector class to Python
    py::class_<GNATVector>(m, "NearestNeighborsGNAT")
        .def(py::init<unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool>(),
             py::arg("degree") = 8,
             py::arg("min_degree") = 4,
             py::arg("max_degree") = 12,
             py::arg("max_num_pts_per_leaf") = 50,
             py::arg("removed_cache_size") = 500,
             py::arg("rebalancing") = false)

        .def("set_distance_function", [](GNATVector &self, py::function py_dist_func) {
            // Create a shared_ptr to store the Python callable
            auto py_dist_func_ptr = std::make_shared<py::function>(py_dist_func);
            // Set the distance function to a lambda that calls the Python callable
            self.setDistanceFunction([py_dist_func_ptr](const pointPtr& a, const pointPtr& b) -> double {
                // Acquire the GIL before calling Python code
                py::gil_scoped_acquire acquire;
                // Convert pointPtr to Python lists
                py::list py_a;
                for(const auto& val : *a)
                    py_a.append(val);
                py::list py_b;
                for(const auto& val : *b)
                    py_b.append(val);
                // Call the Python distance function
                py::object result = (*py_dist_func_ptr)(py_a, py_b);
                // Convert the result to double
                double distance = result.cast<double>();
                return distance;
            });
        })
        .def("set_seed", &GNATVector::setSeed)
        
        // Modified 'add' function to return the pointer
        .def("add", [](GNATVector &self, py::object data) -> std::shared_ptr<point> {
            // Convert the Python object to PointPtr
            auto ptr = convert_to_point_ptr(data);
            self.add(ptr);
            return ptr; // Return the shared_ptr to Python
        }, py::arg("data"))

        // Modified 'add_list' function to return a list of pointers
        .def("add_list", [](GNATVector &self, py::object data_list) -> std::vector<std::shared_ptr<point>> {
            // Convert the Python object to vector<PointPtr>
            auto data_vec = convert_to_point_ptr_list(data_list);
            self.add(data_vec);
            return data_vec; // Return the list of shared_ptr to Python
        }, py::arg("data_list"))

        // Modified 'remove' function to take a pointer
        .def("remove", [](GNATVector &self, std::shared_ptr<point> ptr) -> bool {
            return self.remove(ptr);
        }, py::arg("point_ptr"), "Remove a point using its pointer/handle.")

        .def("clear", &GNATVector::clear)
        .def("size", &GNATVector::size)
        .def("nearest", [](GNATVector &self, py::object data) {
            // Convert to pointPtr
            auto ptr = convert_to_point_ptr(data);
            auto result = self.nearestIndex(ptr);
            unsigned index = std::get<0>(result);
            auto nearest_data_ptr = std::get<1>(result);
            if (!nearest_data_ptr)
                throw std::runtime_error("No nearest neighbor found.");
            // Convert nearest to numpy array
            py::array_t<double> nearest_data(nearest_data_ptr->size());
            std::memcpy(nearest_data.mutable_data(), nearest_data_ptr->data(), nearest_data_ptr->size() * sizeof(double));
            return py::make_tuple(index, nearest_data);
        }, py::arg("data"))
        .def("nearest_k", [](GNATVector &self, py::object data, size_t k) {
            // Convert to pointPtr
            auto ptr = convert_to_point_ptr(data);
            std::vector<pointPtr> neighbors;
            std::vector<unsigned> indices;
            self.nearestKIndices(ptr, k, indices, neighbors);
            // Convert to Python list of numpy arrays and indices
            py::list py_neighbors;
            for(auto &n : neighbors)
            {
                py::array_t<double> neighbor(n->size());
                std::memcpy(neighbor.mutable_data(), n->data(), n->size() * sizeof(double));
                py_neighbors.append(neighbor);
            }
            py::list py_indices;
            for(auto &i : indices)
                py_indices.append(i);
            return py::make_tuple(py_indices, py_neighbors);
        }, py::arg("data"), py::arg("k"))
        .def("nearest_r", [](GNATVector &self, py::object data, double radius) {
            // Convert to pointPtr
            auto ptr = convert_to_point_ptr(data);
            std::vector<pointPtr> neighbors;
            std::vector<unsigned> indices;
            self.nearestRIndices(ptr, radius, indices, neighbors);
            // Convert to Python list of numpy arrays and indices
            py::list py_neighbors;
            for(auto &n : neighbors)
            {
                py::array_t<double> neighbor(n->size());
                std::memcpy(neighbor.mutable_data(), n->data(), n->size() * sizeof(double));
                py_neighbors.append(neighbor);
            }
            py::list py_indices;
            for(auto &i : indices)
                py_indices.append(i);
            return py::make_tuple(py_indices, py_neighbors);
        }, py::arg("data"), py::arg("radius"))
        .def("save", [](const GNATVector &self, const std::string &filename) {
            std::ofstream ofs(filename);
            if (!ofs)
            {
                throw std::runtime_error("Could not open file for writing.");
            }
            ofs << self;
            ofs.close();
        }, py::arg("filename"))
        .def("load", [](GNATVector &self, const std::string &filename) {
            std::ifstream ifs(filename);
            if (!ifs)
            {
                throw std::runtime_error("Could not open file for reading.");
            }
            ifs >> self;
            ifs.close();
        }, py::arg("filename"))
        .def("list", [](GNATVector &self) {
            std::vector<pointPtr> data;
            self.list(data);
            // Convert to Python list of numpy arrays
            py::list py_data;
            for(auto &d : data)
            {
                py::array_t<double> arr(d->size());
                std::memcpy(arr.mutable_data(), d->data(), d->size() * sizeof(double));
                py_data.append(arr);
            }
            return py_data;
        });
}