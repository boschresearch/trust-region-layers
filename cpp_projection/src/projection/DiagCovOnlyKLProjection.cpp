// This source code is from the private project ITPAL
// Copyright (c) 2020-2021 ALRhub
// This source code is licensed under the MIT license found in the
// 3rd-party-licenses.txt file in the root directory of this source tree.

#include <projection/DiagCovOnlyKLProjection.h>

DiagCovOnlyKLProjection::DiagCovOnlyKLProjection(uword dim, int max_eval) :
    dim(dim),
    max_eval(max_eval)
{
    omega_offset = 1.0;  // Default value, might change due to rescaling!
}

vec DiagCovOnlyKLProjection::forward(double eps, const vec &old_var, const vec &target_var){
    this->eps = eps;
    succ = false;

    /** Prepare **/
    old_prec = 1.0 / old_var;
    old_chol_prec = sqrt(old_prec);

    target_prec = 1.0 / target_var;

    old_logdet = - 2 * sum(log(old_chol_prec + 1e-25));

    kl_const_part = old_logdet - dim;

    /** Otpimize **/
    nlopt::opt opt(nlopt::LD_LBFGS, 1);

    opt.set_min_objective([](const std::vector<double> &eta, std::vector<double> &grad, void *instance){
        return ((DiagCovOnlyKLProjection *) instance)->dual(eta, grad);}, this);

    std::vector<double> opt_eta_omega;

    std::tie(succ, opt_eta_omega) = NlOptUtil::opt_dual_1lp(opt, 0.0, max_eval);
    if (!succ) {
        opt_eta_omega[0] = eta;
        succ = NlOptUtil::valid_despite_failure(opt_eta_omega, grad);
    }

    /** Post process**/
    if (succ) {
        eta = opt_eta_omega[0];

        projected_prec = (eta * old_prec + target_prec) / (eta + omega_offset);
        projected_var = 1.0 / projected_prec;
    } else{
        throw std::logic_error("NLOPT failure");
    }
    return projected_var;
}

vec DiagCovOnlyKLProjection::backward(const vec &d_cov) {
    /** takes derivatives of loss w.r.t to projected mean and covariance and propagates back through optimization
      yielding derivatives of loss w.r.t to target mean and covariance **/
    if (!succ){
        throw std::exception();
    }
    /** Prepare **/

    vec deta_dQ_target;
    deta_dQ_target = last_eta_grad();

    double eo = omega_offset + eta;
    double eo_squared = eo * eo;
    vec dQ_deta = (omega_offset * old_prec - target_prec) / eo_squared;

    vec d_Q = - projected_var % d_cov % projected_var;

    double d_eta = sum(d_Q % dQ_deta);

    vec d_Q_target = d_eta * deta_dQ_target + d_Q / eo;

    vec d_cov_target = - target_prec % d_Q_target % target_prec;

    return d_cov_target;
}

double DiagCovOnlyKLProjection::dual(std::vector<double> const &eta_omega, std::vector<double> &grad){
    eta = eta_omega[0] > 0.0 ? eta_omega[0] : 0.0;
    vec new_prec = (eta * old_prec + target_prec) / (eta + omega_offset);
    try {
        /** dual **/
        vec new_var = 1.0 / new_prec;
        vec new_chol_var = sqrt(new_var);
        double new_logdet = 2 * sum(log(new_chol_var) + 1e-25);

        double dual = eta * eps - 0.5 * eta * old_logdet;
        dual += 0.5 * (eta + omega_offset) * new_logdet;

        /** gradient **/
        double trace_term = accu(square(old_chol_prec % new_chol_var));

        double kl = 0.5 * (kl_const_part - new_logdet + trace_term);
        grad[0] = eps - kl;
        this->grad[0] = grad[0];

        return dual;
    } catch (std::runtime_error &err) {
        grad[0] = -1.0;
        this->grad[0] = grad[0];
        return 1e12;
    }
}

vec DiagCovOnlyKLProjection::last_eta_grad() const {

    /** case 1, constraint inactive **/
    if(eta == 0.0) {
        return vec(dim, fill::zeros);

    /** case 2, constraint active **/
    }  else if(eta > 0.0){
        vec dQ_deta = (omega_offset * old_prec - target_prec) / (eta + omega_offset);

        vec tmp = vec(dim, fill::ones) - old_prec % projected_var;
        vec f2_dQ = projected_var % tmp;

        double c = - 1  / sum(f2_dQ % dQ_deta);
        return c * f2_dQ;

    } else {
        throw std::exception();
    }
}
