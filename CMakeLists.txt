project(mschol LANGUAGES CXX C CUDA Fortran)
cmake_minimum_required(VERSION 3.0)
cmake_policy(SET CMP0074 NEW)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/;${CMAKE_MODULE_PATH}")

# Python & torch
find_package(Python REQUIRED COMPONENTS Development)
find_package(Torch REQUIRED)

# BOOST
find_package(Boost REQUIRED COMPONENTS filesystem system iostreams)
if(Boost_FOUND)
  message("-- Boost @ ${Boost_INCLUDE_DIRS}")
  message("-- Boost libs ${Boost_LIBRARIES}")
  include_directories(${Boost_INCLUDE_DIRS})
endif(Boost_FOUND)

# SuiteSparse
find_package(SuiteSparse REQUIRED)
if(SuiteSparse_FOUND)
  message("-- SuiteSparse @ ${SuiteSparse_INCLUDE_DIRS}")
  message("-- SuiteSparse libs @ ${SuiteSparse_LIBRARIES}")
  include_directories(${SuiteSparse_INCLUDE_DIRS})
endif(SuiteSparse_FOUND)

# CUDA runtime libs
find_package(CUDAToolkit REQUIRED)
find_package(Eigen3 REQUIRED)

# HEADER-ONLY LIBS
include_directories(external/eigen/)
include_directories(external/cuda-samples/Common)

include_directories(${PROJECT_SOURCE_DIR})

# C++17 is for compiling Hlibpro
set(CMAKE_CXX_FLAGS "-O3 -Wall -fopenmp -std=c++17 -fpermissive")
add_subdirectory(src)
add_subdirectory(mfs_torch/csrc)