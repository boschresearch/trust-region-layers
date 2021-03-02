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

from typing import Tuple

import numpy as np
import torch as ch
import torch.nn as nn

from trust_region_projections.models.policy.abstract_gaussian_policy import AbstractGaussianPolicy
from trust_region_projections.models.value.vf_net import VFNet
from trust_region_projections.utils.network_utils import initialize_weights
from trust_region_projections.utils.torch_utils import diag_bijector, fill_triangular, fill_triangular_inverse


class GaussianPolicyFull(AbstractGaussianPolicy):
    """
    A Gaussian policy using a fully connected neural network.
    The parameterizing tensor is a mean vector and a cholesky matrix.
    """

    def _get_std_parameter(self, action_dim: int):
        # std = inverse_softplus(ch.ones(action_dim) * init_std).diagflat()
        # flat_chol = fill_triangular_inverse(std)
        chol_shape = action_dim * (action_dim + 1) // 2
        flat_chol = ch.normal(0, 0.01, (chol_shape,))
        return nn.Parameter(flat_chol)

    def _get_std_layer(self, prev_size: int, action_dim: int, init: str):
        chol_shape = action_dim * (action_dim + 1) // 2
        flat_chol = nn.Linear(prev_size, chol_shape)
        initialize_weights(flat_chol, init, scale=0.01)
        return flat_chol

    def forward(self, x: ch.Tensor, train: bool = True):
        self.train(train)

        for affine in self._affine_layers:
            x = self.activation(affine(x))

        flat_chol = self._pre_std(x) if self.contextual_std else self._pre_std
        chol = fill_triangular(flat_chol).expand(x.shape[0], -1, -1)
        chol = diag_bijector(lambda z: self.diag_activation(z + self._pre_activation_shift) + self.minimal_std, chol)

        return self._mean(x), chol

    def sample(self, p: Tuple[ch.Tensor, ch.Tensor], n=1) -> ch.Tensor:
        return self.rsample(p, n).detach()

    def rsample(self, p: Tuple[ch.Tensor, ch.Tensor], n=1) -> ch.Tensor:
        means, chol = p
        eps = ch.randn((n,) + means.shape).to(dtype=chol.dtype, device=chol.device)[..., None]
        samples = (chol @ eps).squeeze(-1) + means
        return samples.squeeze(0)

    def log_probability(self, p: Tuple[ch.Tensor, ch.Tensor], x: ch.Tensor, **kwargs):
        mean, std = p
        k = mean.shape[-1]

        logdet = self.log_determinant(std)
        mean_diff = self.maha(x, mean, std)
        nll = 0.5 * (k * np.log(2 * np.pi) + logdet + mean_diff)
        return -nll

    def entropy(self, p: Tuple[ch.Tensor, ch.Tensor]):
        _, std = p
        k = std.shape[-1]

        logdet = self.log_determinant(std)
        return .5 * (k * np.log(2 * np.e * np.pi) + logdet)

    def log_determinant(self, std: ch.Tensor):
        """
        Returns the log determinant of a cholesky matrix
        Args:
             std: a cholesky matrix
        Returns:
            The determinant of mat, aka product of the diagonal
        """
        return 2 * std.diagonal(dim1=-2, dim2=-1).log().sum(-1)

    def maha(self, mean: ch.Tensor, mean_other: ch.Tensor, std: ch.Tensor):
        diff = (mean - mean_other)[..., None]
        return ch.triangular_solve(diff, std, upper=False)[0].pow(2).sum([-2, -1])

    def precision(self, std: ch.Tensor):
        return ch.cholesky_solve(ch.eye(std.shape[-1], dtype=std.dtype, device=std.device), std, upper=False)

    def covariance(self, std: ch.Tensor):
        std = std.view((-1,) + std.shape[-2:])
        return (std @ std.permute(0, 2, 1)).squeeze(0)

    def set_std(self, std: ch.Tensor) -> None:
        std = diag_bijector(lambda z: self.diag_activation_inv(z - self.minimal_std) - self._pre_activation_shift, std)
        assert self._pre_std.shape == fill_triangular_inverse(std).shape
        self._pre_std.data = fill_triangular_inverse(std)
