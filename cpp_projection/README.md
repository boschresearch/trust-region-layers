# C++ Projection

C++ implementation and python bindings of the numeric KL projection. 
It contains a tuned implementation for the non-batched projection (`DiagCovOnlyProjection.cpp/h`) and the batched-projection (`BatchedDiagCovOnlyProjection.cpp/h`). The latter is parallelizing the former with OpenMP.
The interface to python is provided by pybind11 (`PyProject.cpp`), the conversion for numpy arrays to Armadillo `vec/mat/cube` is provided in `PyArmaConverter.h`.
    
Note that the C++ part uses column-major layout while the python part uses row-major (nothing you need to care about if you just want to use it).

## Setup python
Tested with python 3.6.8 

## Installation of required packages and libraries 
The following libraries are required: gcc, openmp, gfortran, openblas, lapack, and cmake. Install them using your package manager
(On Ubuntu: `sudo apt-get install gcc gfortran libopenblas-dev liblapack-dev cmake `) .
     
### Installation of NLOPT (tested with version 2.6)
Dowload [NLOPT](https://nlopt.readthedocs.io/en/latest/) and follow the installation instructions.
You do not need to install using sudo but can put the library anywhere.

Change the `include_directories` and `link_directories` statements in `cpp_projection/CMakeLists.txt` such that they point to the NLOPT headers and libraries. 

### Install Armadillo (tested with version 9.8000)

Download [Armadillo](http://arma.sourceforge.net/download.html), unpack, and run `./configure`. You do not need to build Armadillo.
Change the `include_directories` and statement in `cpp_projection/CMakeLists.txt` such that it points to the Armadillo headers. 

### Download pybind11 (tested with version 2.4)

Download [pybind11](https://github.com/pybind/pybind11/releases), unpack, rename to pybind11, and place the pybind11 folder under `cpp_projection/`.
You can put it at a different location or install pybind11 using pip, conda, or any other method, but then the `cpp_projection/CMakeLists.txt` needs to be adpated such that pybind11 is found.

### Install CppProjection package 
With your virtualenv being active, go to `cpp_projection` and run 
```
python3 setup.py install --user
```
