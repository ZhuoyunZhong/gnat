cmake_minimum_required(VERSION 3.12)
project(gnat)

# Use the FindBoost provided by CMake, rather than the one provided by Boost
# (for CMake >=3.30).
if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.30")
    cmake_policy(SET CMP0167 OLD)
endif()

include(GNUInstallDirs)  # For standard installation directories

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find pybind11
execute_process(
    COMMAND python3 -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_CMAKE_DIR})
find_package(pybind11 REQUIRED)

set(Boost_USE_MULTITHREADED ON)
find_package(Boost REQUIRED COMPONENTS serialization filesystem system program_options)

find_package(Eigen3 REQUIRED)

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
