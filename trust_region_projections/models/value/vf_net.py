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

from typing import Sequence

import torch.nn as nn

from trust_region_projections.utils.network_utils import get_activation, get_mlp, initialize_weights


class VFNet(nn.Module):

    def __init__(self, obs_dim: int, output_dim: int = 1, init: str = "orthogonal",
                 hidden_sizes: Sequence[int] = (64, 64), activation: str = "tanh"):
        """
        A value network using a fully connected neural network.
        Args:
            obs_dim: Observation dimensionality aka input dimensionality
            output_dim: Action dimensionality aka output dimensionality, generally this is 1
            init: Initialization of layers
            hidden_sizes: Sequence of hidden layer sizes for each hidden layer in the neural network.
            activation: Type of ctivation for hidden layers

        Returns:

        """

        super().__init__()
        self.activation = get_activation(activation)
        self._affine_layers = get_mlp(obs_dim, hidden_sizes, init, True)

        self.final = self.get_final(hidden_sizes[-1], output_dim, init)

    def get_final(self, prev_size, output_dim, init):
        final = nn.Linear(prev_size, output_dim)
        initialize_weights(final, init, scale=1.0)
        return final

    def forward(self, x, train=True):
        """
        Forward pass of the value network
        Args:
            x: States to compute the value estimate for.
        Returns:
            The value of the states x
        """

        self.train(train)

        for affine in self._affine_layers:
            x = self.activation(affine(x))
        return self.final(x).squeeze(-1)
