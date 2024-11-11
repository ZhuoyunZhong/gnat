cmake_minimum_required(VERSION 3.14...3.26)
project(gnat)

include(GNUInstallDirs)  # For standard installation directories

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find pybind11
find_package(pybind11 REQUIRED)

# Find Eigen3
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Find Boost
find_package(Boost REQUIRED)

# Include directories
include_directories(
    ${EIGEN3_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}
)

# Source files
set(SOURCES
    ompl/util/src/RandomNumbers.cpp
    bindings/py_gnat.cpp
)

# Create the Python module
pybind11_add_module(_gnat MODULE ${SOURCES})

# Install the module into the 'gnat' package directory
install(TARGETS _gnat
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX})
