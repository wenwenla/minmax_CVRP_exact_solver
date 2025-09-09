cmake_minimum_required(VERSION 3.12)
project(splitting_solver)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

set(pybind11_DIR "/usr/local/lib/python3.10/dist-packages/pybind11/share/cmake/pybind11")
find_package(pybind11 REQUIRED)

pybind11_add_module(dp_solver mmcvrp_dp.cpp)
