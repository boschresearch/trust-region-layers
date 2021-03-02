// This source code is from the private project ITPAL
// Copyright (c) 2020-2021 ALRhub
// This source code is licensed under the MIT license found in the
// 3rd-party-licenses.txt file in the root directory of this source tree.

#ifndef GENERATOR_NLOPTUTIL_H
#define GENERATOR_NLOPTUTIL_H

#define ARMA_DONT_PRINT_ERRORS

#include <nlopt.hpp>
#include <tuple>

class NlOptUtil{
public:

    static std::tuple<bool, std::vector<double> > opt_dual_2lp(
            nlopt::opt& opt, double lower_bound_eta, double lower_bound_omega, int max_eval){
        /* setting max_eval, even is the number much higher than the number of needed iterations speeds up the
         * optimization by a lot (~ 2 orders of magnitude) for some reason ("compiler magic :-)" ).
         * - The number can be adapted at runtime
         * - This throws an error if the max_number of iterations is actually needed
         * - I took it to the extrem and the speed up even works with max_eval=1000, usually less than 20 evals are
         *   needed for the dual
         * - If anyone finds this and has insights to why this works, please write me an email (philipp.becker@kit.edu)
         */
        std::vector<double> lower_bound(2);
        lower_bound[0] = lower_bound_eta;
        lower_bound[1] = lower_bound_omega;
        opt.set_lower_bounds(lower_bound);
        opt.set_upper_bounds(1e12);
        opt.set_maxeval(max_eval);
        
        std::vector<double> eta_omega = std::vector<double>(2, 10.0);
        double dual_value;
        nlopt::result res;
        try{
            res = opt.optimize(eta_omega, dual_value);
        } catch (std::exception &ex){
            res = nlopt::FAILURE;
        }
        if (opt.get_numevals() >= max_eval){
            res = nlopt::FAILURE;
        }
        return std::make_tuple(res > 0, eta_omega);
    }

    static std::tuple<bool, std::vector<double> > opt_dual_1lp(nlopt::opt& opt, double lower_bound_eta, int max_eval){
        /* max eval, same logic as above */
        std::vector<double> lower_bound(1);
        lower_bound[0] = lower_bound_eta;
        opt.set_lower_bounds(lower_bound);
        opt.set_upper_bounds(1e12);
        opt.set_maxeval(max_eval);

        std::vector<double> eta = std::vector<double>(1, 1.0);
        double dual_value;
        nlopt::result res;
        try{
            res = opt.optimize(eta, dual_value);
        } catch (std::exception &ex){
            res = nlopt::FAILURE;
        }
        if (opt.get_numevals() >= max_eval){
            res = nlopt::FAILURE;
        }
        return std::make_tuple(res > 0, eta);
    }

    static bool valid_despite_failure(std::vector<double>& lp, std::vector<double>& grad){
        /*NLOPT sometimes throws errors because the dual and gradients do not fit together anymore for numerical reasons
         * This problem becomes severe for high dimensional data.
         * However, that happens mostly after the algorithm is almost converged. We check for those instances and just
         * work with the last values.
         *
         */
        //gradient norm close to 0
        double grad_bound = 1e-5;
        double value_bound = 1e-10;
        if (sqrt(grad[0] * grad[0]) < grad_bound){
            return true;
        }
        //omega at lower bound and gradient for eta close to 0
        if (lp[0] < value_bound){
            return true;
        }
        return false;
    }
};


#endif //GENERATOR_NLOPTUTIL_H
