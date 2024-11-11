#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>   // For automatic STL container bindings
#include <pybind11/numpy.h> // For NumPy array support (if needed)
#include "ompl/datastructures/NearestNeighborsGNAT.h"

namespace py = pybind11;
namespace ompl = ::ompl; // Ensure the ompl namespace is accessible

// Define the type you will use with the GNAT.
// For example, using a vector of doubles to represent a point in space.
using Point = std::vector<double>;

// Create an alias for the NearestNeighborsGNAT instantiated with Point
using NearestNeighborsGNATPoint = ompl::NearestNeighborsGNAT<Point>;

PYBIND11_MODULE(_gnat, m)
{
    py::class_<NearestNeighborsGNATPoint>(m, "NearestNeighborsGNAT")
        .def(py::init<>())
        .def("set_distance_function", &NearestNeighborsGNATPoint::setDistanceFunction)
        .def("add", (void(NearestNeighborsGNATPoint::*)(const Point &)) & NearestNeighborsGNATPoint::add)
        .def("add_list", (void(NearestNeighborsGNATPoint::*)(const std::vector<Point> &)) & NearestNeighborsGNATPoint::add)
        .def("nearest", &NearestNeighborsGNATPoint::nearest)
        .def("nearest_k", [](NearestNeighborsGNATPoint &self, const Point &data, std::size_t k) {
            std::vector<Point> nbh;
            self.nearestK(data, k, nbh);
            return nbh; })
        .def("nearest_r", [](NearestNeighborsGNATPoint &self, const Point &data, double radius) {
            std::vector<Point> nbh;
            self.nearestR(data, radius, nbh);
            return nbh; })
        .def("remove", &NearestNeighborsGNATPoint::remove)
        .def("clear", &NearestNeighborsGNATPoint::clear)
        .def("size", &NearestNeighborsGNATPoint::size)
        .def("list", [](NearestNeighborsGNATPoint &self) {
            std::vector<Point> data;
            self.list(data);
            return data; });
}
