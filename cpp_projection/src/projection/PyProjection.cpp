// This source code is from the private project ITPAL
// Copyright (c) 2020-2021 ALRhub
// This source code is licensed under the MIT license found in the
// 3rd-party-licenses.txt file in the root directory of this source tree.

#include <pybind11/pybind11.h>
#include <PyArmaConverter.h>

#include <projection/DiagCovOnlyKLProjection.h>
#include <projection/BatchedDiagCovOnlyProjection.h>

namespace py = pybind11;

PYBIND11_MODULE(cpp_projection, p){

    /* ------------------------------------------------------------------------------
    DIAG COVAR ONLY PROJECTION
    --------------------------------------------------------------------------------*/
    py::class_<DiagCovOnlyKLProjection> dcop(p, "DiagCovOnlyKLProjection");

    dcop.def(py::init([](uword dim, int max_eval){return new DiagCovOnlyKLProjection(dim, max_eval);}),
           py::arg("dim"), py::arg("max_eval") = 100);

    dcop.def("forward", [](DiagCovOnlyKLProjection* obj, double eps, dpy_arr old_covar, dpy_arr target_covar){
               return from_mat<double>(obj->forward(eps, to_vec<double>(old_covar), to_vec<double>(target_covar)));},
           py::arg("eps"),py::arg("old_covar"), py::arg("target_covar"));

    dcop.def("backward", [](DiagCovOnlyKLProjection* obj, dpy_arr dl_dcovar_projected){
               return from_mat<double>(obj->backward(to_vec<double>(dl_dcovar_projected)));},
      py::arg("dl_dcovar_projected"));

    dcop.def_property_readonly("last_eta", &DiagCovOnlyKLProjection::get_last_eta);
    dcop.def_property_readonly("was_succ", &DiagCovOnlyKLProjection::was_succ);

    /* ------------------------------------------------------------------------------
    BATCHED DIAG COVAR ONLY PROJECTION

    --------------------------------------------------------------------------------*/
    py::class_<BatchedDiagCovOnlyProjection> bdcop(p, "BatchedDiagCovOnlyProjection");
    bdcop.def(py::init([](uword batch_size, uword dim, int max_eval){
        return new BatchedDiagCovOnlyProjection(batch_size, dim, max_eval);}),
           py::arg("batchsize"), py::arg("dim"), py::arg("max_eval") = 100);

    bdcop.def("forward", [](BatchedDiagCovOnlyProjection* obj, dpy_arr epss, dpy_arr old_vars,
            dpy_arr target_vars){
           try {
                   mat vars = obj->forward(to_vec<double>(epss), to_mat<double>(old_vars), to_mat<double>(target_vars));
                   return from_mat<double>(vars);
               } catch (std::invalid_argument &e) {
                   PyErr_SetString(PyExc_AssertionError, e.what());
               }
           },
           py::arg("epss"),py::arg("old_var"), py::arg("target_var")
    );

    bdcop.def("backward", [](BatchedDiagCovOnlyProjection* obj, dpy_arr d_vars){
               mat d_vars_d_target = obj->backward(to_mat<double>(d_vars));
               return from_mat<double>(d_vars_d_target);}, py::arg("d_vars"));

}