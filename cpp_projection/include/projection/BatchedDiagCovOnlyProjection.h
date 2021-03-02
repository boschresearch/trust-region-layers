// This source code is from the private project ITPAL
// Copyright (c) 2020-2021 ALRhub
// This source code is licensed under the MIT license found in the
// 3rd-party-licenses.txt file in the root directory of this source tree.

#ifndef IPP_BATCHEDDIAGCOVONLYPROJECTION_H
#define IPP_BATCHEDDIAGCOVONLYPROJECTION_H

#include <armadillo>
#include <nlopt.hpp>
#include <NlOptUtil.h>
#include <projection/DiagCovOnlyKLProjection.h>
#include <cblas.h>

using namespace arma;

class BatchedDiagCovOnlyProjection{

public:

    BatchedDiagCovOnlyProjection(uword batch_size, uword dim, int max_eval);

    mat forward(const vec &epss, const mat &old_vars, const mat &target_vars);
    mat backward(const mat &d_vars);

private:

    double kl_diag_cov_only(const vec& cc1, const vec& cc2) const;
    std::vector<DiagCovOnlyKLProjection> projectors;
    uword batch_size, dim;
    std::vector<bool> projection_applied;

};

#endif //IPP_BATCHEDDIAGCOVONLYPROJECTION_H
