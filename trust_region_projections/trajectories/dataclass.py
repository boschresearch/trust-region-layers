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
from typing import NamedTuple


class TrajectoryOnPolicyRaw(NamedTuple):
    obs: ch.Tensor
    actions: ch.Tensor
    logpacs: ch.Tensor
    rewards: ch.Tensor
    values: ch.Tensor
    dones: ch.Tensor
    time_limit_dones: ch.Tensor
    means: ch.Tensor
    stds: ch.Tensor


class TrajectoryOnPolicy(NamedTuple):
    obs: ch.Tensor
    actions: ch.Tensor
    logpacs: ch.Tensor
    rewards: ch.Tensor
    returns: ch.Tensor
    advantages: ch.Tensor
    values: ch.Tensor
    dones: ch.Tensor
    time_limit_dones: ch.Tensor
    q: tuple
