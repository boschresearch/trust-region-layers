#   Copyright (c) 2021 Robert Bosch GmbH
#   Author: Fabian Otto
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Affero General Public License as published
#   by the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Affero General Public License for more details.
#
#   You should have received a copy of the GNU Affero General Public License
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.

import torch as ch
from typing import Tuple

from trust_region_projections.models.policy.abstract_gaussian_policy import AbstractGaussianPolicy
from trust_region_projections.projections.base_projection_layer import BaseProjectionLayer, mean_projection
from trust_region_projections.utils.projection_utils import gaussian_frobenius


class FrobeniusProjectionLayer(BaseProjectionLayer):

    def _trust_region_projection(self, policy: AbstractGaussianPolicy, p: Tuple[ch.Tensor, ch.Tensor],
                                 q: Tuple[ch.Tensor, ch.Tensor], eps: ch.Tensor, eps_cov: ch.Tensor, **kwargs):
        """
        Runs Frobenius projection layer and constructs cholesky of covariance

        Args:
            policy: policy instance
            p: current distribution
            q: old distribution
            eps: (modified) kl bound/ kl bound for mean part
            eps_cov: (modified) kl bound for cov part
            beta: (modified) entropy bound
            **kwargs:
        Returns: mean, cov cholesky
        """

        mean, chol = p
        old_mean, old_chol = q
        batch_shape = mean.shape[:-1]

        ####################################################################################################################
        # precompute mean and cov part of frob projection, which are used for the projection.
        mean_part, cov_part, cov, cov_old = gaussian_frobenius(policy, p, q, self.scale_prec, True)

        ################################################################################################################
        # mean projection maha/euclidean

        proj_mean = mean_projection(mean, old_mean, mean_part, eps)

        ################################################################################################################
        # cov projection frobenius

        cov_mask = cov_part > eps_cov

        if cov_mask.any():
            # alpha = ch.where(fro_norm_sq > eps_cov, ch.sqrt(fro_norm_sq / eps_cov) - 1., ch.tensor(1.))
            eta = ch.ones(batch_shape, dtype=chol.dtype, device=chol.device)
            eta[cov_mask] = ch.sqrt(cov_part[cov_mask] / eps_cov) - 1.
            eta = ch.max(-eta, eta)

            new_cov = (cov + ch.einsum('i,ijk->ijk', eta, cov_old)) / (1. + eta + 1e-16)[..., None, None]
            proj_chol = ch.where(cov_mask[..., None, None], ch.cholesky(new_cov), chol)
        else:
            proj_chol = chol

        return proj_mean, proj_chol

    def trust_region_value(self, policy, p, q):
        """
        Computes the Frobenius metric between two Gaussian distributions p and q.
        Args:
            policy: policy instance
            p: current distribution
            q: old distribution
        Returns:
            mean and covariance part of Frobenius metric
        """
        return gaussian_frobenius(policy, p, q, self.scale_prec)

    def get_trust_region_loss(self, policy: AbstractGaussianPolicy, p: Tuple[ch.Tensor, ch.Tensor],
                              proj_p: Tuple[ch.Tensor, ch.Tensor]):

        mean_diff, _ = self.trust_region_value(policy, p, proj_p)
        if policy.contextual_std:
            # Compute MSE here, because we found the Frobenius norm tends to generate values that explode for the cov
            cov_diff = (p[1] - proj_p[1]).pow(2).sum([-1, -2])
            delta_loss = (mean_diff + cov_diff).mean()
        else:
            delta_loss = mean_diff.mean()

        return delta_loss * self.trust_region_coeff
