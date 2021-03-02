cmake_minimum_required(VERSION 2.8.12)
project(IPP)

add_definitions(-std=c++11)

find_package(OpenMP REQUIRED)
find_library(gfortran REQUIRED)
find_library(openblas REQUIRED)
find_library(lapack REQUIRED)
### FOR INSTALLATION CHANGE STUFF HERE! ###################################

#Add pybind11 subfolder (under cpp) or change command if pybind11 installed on system
add_subdirectory(pybind11)

#Adapt these so that armadillo and nlopt are found
include_directories(<path_to_armadillo>/include <path_to_nlopt>/include)
link_directories(<path_to_nlopt>/lib)
### END OF STUFF THE NEEDS TO BE CHANGED ##################################

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lopenblas")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -llapack")
include_directories(include)

#Projection
add_library(projectionlib SHARED
        src/projection/DiagCovOnlyKLProjection.cpp
        src/projection/BatchedDiagCovOnlyProjection.cpp
        )
target_link_libraries(projectionlib lapack gfortran nlopt)

pybind11_add_module(cpp_projection src/projection/PyProjection.cpp)
target_link_libraries(cpp_projection PUBLIC lapack projectionlib)
