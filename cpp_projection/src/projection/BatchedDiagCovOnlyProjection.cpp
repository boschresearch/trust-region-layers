// This source code is from the private project ITPAL
// Copyright (c) 2020-2021 ALRhub
// This source code is licensed under the MIT license found in the
// 3rd-party-licenses.txt file in the root directory of this source tree.

#include <projection/BatchedDiagCovOnlyProjection.h>

BatchedDiagCovOnlyProjection::BatchedDiagCovOnlyProjection(uword batch_size, uword dim, int max_eval) :
    batch_size(batch_size),
    dim(dim){
    for (int i = 0; i < batch_size; ++i) {
        projectors.emplace_back(DiagCovOnlyKLProjection(dim, max_eval));
        projection_applied.emplace_back(false);
    }
    openblas_set_num_threads(1);
}

mat BatchedDiagCovOnlyProjection::forward(const vec &epss, const mat &old_vars, const mat &target_vars) {

    mat vars(size(old_vars));
    bool failed = false;
    std::stringstream stst;

    #pragma omp parallel for default(none) schedule(static) shared(epss, old_vars, target_vars, vars, failed, stst)
    for (int i = 0; i < batch_size; ++i) {
        double eps = epss.at(i);
        const vec &old_var = old_vars.col(i);
        const vec &target_var = target_vars.col(i);

        vec occ = sqrt(old_var);
        vec tcc = sqrt(target_var);
        double kl_ = kl_diag_cov_only(tcc, occ);

        if (kl_ <= eps) {
            vars.col(i) = target_var;
            projection_applied.at(i) = false;
        } else {
            try {
                vec var = projectors[i].forward(eps, old_var, target_var);
                vars.col(i) = var;
                projection_applied.at(i) = true;
            } catch (std::logic_error &e) {
                stst << "Failure during projection " << i << ": " << e.what() << " ";
                failed = true;
            }
          
        }
    }
    if (failed) {
        throw std::invalid_argument(stst.str());
    }
    return vars;
}

mat BatchedDiagCovOnlyProjection::backward(const mat &d_vars) {
    mat d_vars_target(size(d_vars));

    #pragma omp parallel for default(none) schedule(static) shared(d_vars, d_vars_target)
    for (int i = 0; i < batch_size; ++i) {
        vec d_var = d_vars.col(i);

        d_vars_target.col(i) = projection_applied.at(i) ? projectors[i].backward(d_var) : d_var;
    }
    return d_vars_target;
}


double BatchedDiagCovOnlyProjection::kl_diag_cov_only(const vec &cc1, const vec &cc2) const {
    vec cc2_inv_t = 1.0 / cc2;
    double logdet_term = 2 * (sum(log(cc2 + 1e-25)) - sum(log(cc1 + 1e-25)));
    double trace_term = sum(square(cc2_inv_t % cc1));
    double kl = 0.5 * (logdet_term + trace_term - dim);
    return kl;
}


