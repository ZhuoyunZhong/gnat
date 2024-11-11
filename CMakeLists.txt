cmake_minimum_required(VERSION 3.14...3.26)
project(gnat)

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

# Find Eigen3 and Boost
if(WIN32)
    # Find Eigen path with Chocolatey
    execute_process(
        COMMAND powershell -Command "(Get-ChildItem -Path C:/ProgramData/chocolatey/lib/eigen* -Directory | Select-Object -ExpandProperty FullName)"
        OUTPUT_VARIABLE EIGEN_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    list(APPEND CMAKE_PREFIX_PATH "${EIGEN_PATH}/include")
    # Find Boost path with Chocolatey
    execute_process(
        COMMAND powershell -Command "(Get-ChildItem -Path C:/ProgramData/chocolatey/lib/boost* -Directory | Select-Object -ExpandProperty FullName)"
        OUTPUT_VARIABLE BOOST_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    list(APPEND CMAKE_PREFIX_PATH "${BOOST_PATH}/include")
elseif(APPLE)
    # macOS: Use Homebrew to find Eigen and Boost
    execute_process(
        COMMAND brew --prefix eigen
        OUTPUT_VARIABLE EIGEN3_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    execute_process(
        COMMAND brew --prefix boost
        OUTPUT_VARIABLE BOOST_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    list(APPEND CMAKE_PREFIX_PATH ${EIGEN3_PREFIX} ${BOOST_PREFIX})
elseif(UNIX)
    # Linux: Assume default paths for system-installed libraries
    list(APPEND CMAKE_PREFIX_PATH "/usr/share/eigen3/cmake" "/usr/include/boost")
endif()
find_package(Eigen3 REQUIRED)
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
