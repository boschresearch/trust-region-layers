// This source code is from the private project ITPAL
// Copyright (c) 2020-2021 ALRhub
// This source code is licensed under the MIT license found in the
// 3rd-party-licenses.txt file in the root directory of this source tree.

#ifndef GENERATOR_PYARMACONVERTER_H
#define GENERATOR_PYARMACONVERTER_H


#include <pybind11/numpy.h>
#include <armadillo>

/*** CAVEAT: All Matrices  (and Cubes) are implicitly transposed when going from c++ to python and vice versa.
 *  This is done to account for armadillos column-major and numpys/pytorchs row-major storage without losing performance
 *  CAVEAT2: I check more or less nothing while parsing - so some caution is required by the user
 ***/


using namespace arma;
namespace py = pybind11;

template <typename d_type>
using py_c_array = py::array_t<d_type, py::array::c_style || py::array::forcecast>;

typedef py_c_array<double> dpy_arr;

template <typename d_type> Col<d_type> to_vec(py_c_array<d_type> py_vec){
    py::buffer_info info = py_vec.request();
    if(info.itemsize != sizeof(d_type)){
        throw std::runtime_error("Size of matrix entries not equal size of requested data type");
    }
    unsigned long shape_size = info.shape.size();
    switch (shape_size){
        case 1:
            return Col<d_type>((d_type *)info.ptr, (uword) info.shape[0], false, true);
        case 2:
            if (info.shape[1] == 1){
                return Col<d_type>((d_type *)info.ptr, (uword) info.shape[0], false, true);
            } else {
                throw std::logic_error("For dim 2 call to_mat");
            }
        case 3:
            throw std::logic_error("For dim 3 call to_cube");
        default:
            throw std::range_error("Nd Arrays with dim > 3 not supported");
    };
}

template <typename d_type> Row<d_type> to_rowvec(py_c_array<d_type> py_vec){
    py::buffer_info info = py_vec.request();
    if(info.itemsize != sizeof(d_type)){
        throw std::runtime_error("Size of matrix entries not equal size of requested data type");
    }
    unsigned long shape_size = info.shape.size();
    switch (shape_size){
        case 1:
            return Row<d_type>((d_type *)info.ptr, (uword) info.shape[0], false, true);
        case 2:
            if (info.shape[1] == 1){
                return Row<d_type>((d_type *)info.ptr, (uword) info.shape[0], false, true);
            } else {
                throw std::logic_error("For dim 2 call to_mat");
            }
        case 3:
            throw std::logic_error("For dim 3 call to_cube");
        default:
            throw std::range_error("Nd Arrays with dim > 3 not supported");
    };
}

template <typename d_type> Mat<d_type> to_mat(py_c_array<d_type>  py_mat){
    py::buffer_info info = py_mat.request();
    if(info.itemsize != sizeof(d_type)){
        throw std::runtime_error("Size of matrix entries not equal size of requested data type");
    }
    unsigned long shape_size = info.shape.size();
    switch (shape_size){
    case 1:
        throw std::logic_error("For dim 1 call to_vec");
    case 2:
        return Mat<d_type>((d_type *)info.ptr, (uword) info.shape[1], (uword) info.shape[0], false, true);
    case 3:
        throw std::logic_error("For dim 3 call to_cube");
    default:
        throw std::range_error("Nd Arrays with dim > 3 not supported");
    };
}

template <typename d_type> Cube<d_type> to_cube(py_c_array<d_type>  py_cube){
    py::buffer_info info = py_cube.request();
    if(info.itemsize != sizeof(d_type)){
        throw std::runtime_error("Size of matrix entries not equal size of requested data type");
    }
    unsigned long shape_size = info.shape.size();
    switch (shape_size){
        case 1:
            throw std::logic_error("For dim 1 call to_vec");
        case 2:
            throw std::logic_error("For dim 1 call to_mat");
        case 3:
            return Cube<d_type>((d_type *) info.ptr, (uword) info.shape[2], (uword) info.shape[1], (uword) info.shape[0],
                                false, true);
        default:
            throw std::range_error("Nd Arrays with dim > 3 not supported");
    };
}


template <typename d_type> std::vector<Col<d_type> > to_vecs(std::vector<py_c_array<d_type> >py_vecs){
    std::vector<Col<d_type> > c_mats;
    for(uword i = 0; i < py_vecs.size(); ++i){
        c_mats.emplace_back(to_vec<d_type>(py_vecs[i]));
    }
    return c_mats;
}

template <typename d_type> std::vector<Mat<d_type> > to_mats(std::vector<py_c_array<d_type> > py_mats){
    std::vector<Mat<d_type > > c_mats;
    for(uword i = 0; i < py_mats.size(); ++i){
        c_mats.push_back(to_mat<d_type>(py_mats[i]));
    }
    return c_mats;
}

template <typename d_type> py::array_t<d_type> from_mat (Mat<d_type > c_mat){
    if (c_mat.n_cols == 1){
        return py::array_t<d_type>(std::vector<uword>({c_mat.n_rows}), c_mat.memptr());
    }
    return py::array_t<d_type>(std::vector<uword>({c_mat.n_cols, c_mat.n_rows}), c_mat.memptr());
}

template <typename d_type> py::array_t<d_type> from_mat_without_implicit_transpose (Mat<d_type > c_mat){
    mat c_trans = c_mat.t();
    return py::array_t<d_type>(std::vector<uword>({c_mat.n_rows, c_mat.n_cols}), c_trans.memptr());
}



template <typename  d_type> py::array_t<d_type> from_mat_enforce_mat(Mat<d_type> c_mat){
    return py::array_t<d_type>(std::vector<uword>({c_mat.n_cols, c_mat.n_rows}), c_mat.memptr());
};


template <typename d_type> py::array_t<d_type> from_cube (Cube<d_type > c_cube){
    return py::array_t<d_type>(std::vector<uword>({c_cube.n_slices, c_cube.n_cols, c_cube.n_rows}), c_cube.memptr());
}

template <typename d_type> std::tuple<py::array_t<d_type>, py::array_t<d_type> > from_mat(std::tuple<Mat<d_type>, Mat<d_type> >c_mat_t){
    return std::make_tuple(from_mat<d_type>(std::get<0>(c_mat_t)), from_mat_enforce_mat<d_type>(std::get<1>(c_mat_t)));
}

template <typename d_type> std::vector<std::tuple<py::array_t<d_type>, py::array_t<d_type> > > from_mats(std::vector<std::tuple<Mat<d_type>, Mat<d_type> > > c_mat_tuple_vec){
    std::vector<std::tuple<py::array_t<d_type>, py::array_t<d_type> > > py_mat_tuple_vec;
    for(uword i = 0; i < c_mat_tuple_vec.size(); ++i){
        py_mat_tuple_vec.push_back(from_mat<d_type>(c_mat_tuple_vec[i]));
    }
    return py_mat_tuple_vec;
}


template <typename d_type> std::vector<py::array_t<d_type> > from_mats(std::vector<Mat<d_type> > c_mats){
    std::vector<py::array_t<d_type> > py_mats;
    for(uword i = 0; i < c_mats.size(); ++i){
        py_mats.push_back(from_mat<d_type>(c_mats[i]));
    }
    return py_mats;
}

template <typename d_type> std::vector<py::array_t<d_type> > from_cubes(std::vector<Cube<d_type> > c_cubes){
    std::vector<py::array_t<d_type> > py_cubes;
    for(uword i = 0; i < c_cubes.size(); ++i){
        py_cubes.push_back(from_cube<d_type>(c_cubes[i]));
    }
    return py_cubes;
}

#endif //GENERATOR_PYARMACONVERTER_H
