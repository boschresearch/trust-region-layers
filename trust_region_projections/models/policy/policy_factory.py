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

from trust_region_projections.models.policy.gaussian_policy_diag import GaussianPolicyDiag
from trust_region_projections.models.policy.gaussian_policy_full import GaussianPolicyFull
from trust_region_projections.models.policy.gaussian_policy_sqrt import GaussianPolicySqrt


def get_policy_network(policy_type, proj_type, device: ch.device = "cpu", dtype=ch.float32, **kwargs):
    """
    Policy network factory for generating the required Gaussian policy model.
    Args:
        policy_type: 'full' or 'diag' covariance
        proj_type: Which projection is used.
        device: Torch device
        dtype: Torch dtype
        **kwargs: Policy arguments

    Returns:
        Gaussian Policy instance
    """

    if policy_type == "full":
        policy = GaussianPolicySqrt(**kwargs) if "w2" in proj_type else GaussianPolicyFull(**kwargs)
    elif policy_type == "diag":
        policy = GaussianPolicyDiag(**kwargs)
    else:
        raise ValueError(f"Invalid policy type {policy_type}. Select one of 'full', 'diag'.")

    return policy.to(device, dtype)
