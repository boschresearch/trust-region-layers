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
import torch.nn as nn

from trust_region_projections.models.policy.gaussian_policy_full import GaussianPolicyFull
from trust_region_projections.models.value.vf_net import VFNet


class GaussianPolicySqrt(GaussianPolicyFull):
    """
    A Gaussian policy using a fully connected neural network.
    The parameterizing tensor is a mean vector and the true matrix square root of the standard deviation.
    """

    def __init__(self, obs_dim, action_dim, init, hidden_sizes=(64, 64), activation: str = "tanh",
                 contextual_std: bool = False, init_std: float = 1., minimal_std: float = 1e-5, share_weights=False,
                 vf_model: VFNet = None):
        super().__init__(obs_dim, action_dim, init, hidden_sizes, activation, contextual_std, init_std, minimal_std,
                         share_weights, vf_model)
        self.diag_activation = nn.Softplus()

    def forward(self, x: ch.Tensor, train: bool = True):
        mean, chol = super(GaussianPolicySqrt, self).forward(x, train)
        sqrt = chol @ chol.permute(0, 2, 1)

        return mean, sqrt

    def log_determinant(self, std):
        """
        Returns the log determinant of a sqrt matrix
        Args:
            std: sqrt matrix
        Returns:
            The log determinant of std, aka log sum the diagonal

        """
        return 4 * ch.cholesky(std).diagonal(dim1=-2, dim2=-1).log().sum(-1)

    def maha(self, mean, mean_other, std):
        diff = (mean - mean_other)[..., None]
        return (ch.solve(diff, std)[0] ** 2).sum([-2, -1])

    def precision(self, std):
        cov = self.covariance(std)
        return ch.solve(ch.eye(cov.shape[-1], dtype=std.dtype, device=std.device), cov)[0]

    def covariance(self, std: ch.Tensor):
        return std @ std

    def _get_preactivation_shift(self, init_std, minimal_std):
        return self.diag_activation_inv(ch.sqrt(init_std) - ch.sqrt(minimal_std))

    def set_std(self, std: ch.Tensor) -> None:
        std = ch.cholesky(std, upper=False)
        super(GaussianPolicySqrt, self).set_std(std)

    @property
    def is_root(self):
        return True
