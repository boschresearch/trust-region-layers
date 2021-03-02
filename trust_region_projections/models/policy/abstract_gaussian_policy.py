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

from abc import ABC, abstractmethod
from typing import Sequence, Tuple

import torch as ch
import torch.nn as nn

from trust_region_projections.models.value.vf_net import VFNet
from trust_region_projections.utils.network_utils import get_activation, get_mlp, initialize_weights
from trust_region_projections.utils.torch_utils import inverse_softplus


class AbstractGaussianPolicy(nn.Module, ABC):

    def __init__(self, obs_dim: int, action_dim: int, init: str = "orthogonal", hidden_sizes: Sequence[int] = (64, 64),
                 activation: str = "tanh", contextual_std: bool = False, init_std: float = 1.,
                 minimal_std: float = 1e-5, share_weights: bool = False, vf_model: VFNet = None):
        """
        Abstract Method defining a Gaussian policy structure.
        Args:
            obs_dim: Observation dimensionality aka input dimensionality
            action_dim: Action dimensionality aka output dimensionality
            init: Initialization type for the layers 
            hidden_sizes: Sequence of hidden layer sizes for each hidden layer in the neural network.
            activation: Type of ctivation for hidden layers
            contextual_std: Whether to use a contextual standard deviation or not
            init_std: initial value of the standard deviation matrix
            minimal_std: minimal standard deviation
            share_weights: Use joint value and policy network
            vf_model: Optional model when training value and policy model jointly.

        Returns:

        """
        super().__init__()

        self.activation = get_activation(activation)
        self.action_dim = action_dim
        self.contextual_std = contextual_std
        self.share_weights = share_weights
        self.minimal_std = minimal_std
        self.init_std = ch.tensor(init_std)

        self._affine_layers = get_mlp(obs_dim, hidden_sizes, init, True)

        prev_size = hidden_sizes[-1]

        self.diag_activation = nn.Softplus()
        self.diag_activation_inv = inverse_softplus

        # This shift is applied to the Parameter/cov NN output before applying the transformation
        # and gives hence the wanted initial cov
        self._pre_activation_shift = self._get_preactivation_shift(self.init_std, minimal_std)
        self._mean = self._get_mean(action_dim, prev_size, init)
        self._pre_std = self._get_std(contextual_std, action_dim, prev_size, init)

        self.vf_model = vf_model

        if share_weights:
            self.final_value = nn.Linear(prev_size, 1)
            initialize_weights(self.final_value, init, scale=1.0)

    @abstractmethod
    def forward(self, x, train=True):
        pass

    def get_value(self, x, train=True):
        if self.share_weights:
            self.train(train)
            for affine in self.affine_layers:
                x = self.activation(affine(x))
            value = self.final_value(x)
        elif self.vf_model:
            value = self.vf_model(x, train)
        else:
            raise ValueError("Must be sharing weights or use joint training to use get_value.")

        return value

    def squash(self, x):
        """
        Post sampling transformation
        Args: 
            x: values to transform 
        Returns: 
            transformed value
        """
        return x

    def _get_mean(self, action_dim, prev_size=None, init=None, scale=0.01):
        """
        Constructor method for mean prediction.
        Args:
            action_dim: action dimension for output shape
            prev_size: previous layer's output size
            init: initialization type of layer.
            scale

        Returns:
            Mean parametrization.
        """
        mean = nn.Linear(prev_size, action_dim)
        initialize_weights(mean, init, scale=scale)
        return mean

    # @final
    def _get_std(self, contextual_std: bool, action_dim, prev_size=None, init=None):
        """
        Constructor method for std prediction. Do not overwrite.
        Args:
            contextual_std: whether to make the std context dependent or not
            action_dim: action dimension for output shape
            prev_size: previous layer's output size
            init: initialization type of layer.

        Returns:
            Standard deviation parametrization.
        """
        if contextual_std:
            return self._get_std_layer(prev_size, action_dim, init)
        else:
            return self._get_std_parameter(action_dim)

    def _get_preactivation_shift(self, init_std, minimal_std):
        """
        Compute the prediction shift to enforce an initial covariance value for contextual and non contextual policies.
        Args:
            init_std: value to initialize the covariance output with.
            minimal_std: lower bound on the covariance.

        Returns:
            Preactivation shift to enforce minimal and initial covariance.
        """
        return self.diag_activation_inv(init_std - minimal_std)

    @abstractmethod
    def _get_std_parameter(self, action_dim):
        """
        Creates a trainable variable for predicting the std for a non contextual policy.
        Args:
            action_dim: Action dimension for output shape

        Returns:
            Torch trainable variable for covariance prediction.
        """
        pass

    @abstractmethod
    def _get_std_layer(self, prev_size, action_dim, init):
        """
        Creates a layer for predicting the std for a contextual policy.
        Args:
            prev_size: Previous layer's output size
            action_dim: Action dimension for output shape
            init: Initialization type of layer.

        Returns:
            Torch layer for covariance prediction.
        """
        pass

    @abstractmethod
    def sample(self, p: Tuple[ch.Tensor, ch.Tensor], n=1) -> ch.Tensor:
        """
        Given prob dist p=(mean, var), generate samples WITHOUT reparametrization trick
         Args:
            p: Tuple (means, var). means (batch_size, action_space), var (action_space,).
                p are batched probability distributions you're sampling from
            n: Number of samples

        Returns:
            Actions sampled from p_i (batch_size, action_dim)
        """
        pass

    @abstractmethod
    def rsample(self, p: Tuple[ch.Tensor, ch.Tensor], n=1) -> ch.Tensor:
        """
        Given prob dist p=(mean, var), generate samples WITH reparametrization trick.
        This version applies the reparametrization trick to allow for backpropagate through it.
         Args:
            p: Tuple (means, var). means (batch_size, action_space), var (action_space,).
                p are batched probability distributions you're sampling from
            n: Number of samples
        Returns:
            Actions sampled from p_i (batch_size, action_dim)
        """
        pass

    @abstractmethod
    def log_probability(self, p: Tuple[ch.Tensor, ch.Tensor], x: ch.Tensor, **kwargs) -> ch.Tensor:
        """
        Computes the log probability of x given a batched distributions p (mean, std)
        Args:
            p: Tuple (means, var). means (batch_size, action_space), var (action_space,).
            x: Values to compute logpacs for
            **kwargs:

        Returns:
            Log probabilities of x.
        """
        pass

    @abstractmethod
    def entropy(self, p: Tuple[ch.Tensor, ch.Tensor]) -> ch.Tensor:
        """
        Get entropies over the probability distributions given by p = (mean, var).
        mean shape (batch_size, action_space), var shape (action_space,)
        Args: 
            p: Tuple (means, var). means (batch_size, action_space), var (action_space,).
            
        Returns: 
            Policy entropy based on sampled distributions p.
        """
        pass

    @abstractmethod
    def log_determinant(self, std: ch.Tensor) -> ch.Tensor:
        """
        Returns the log determinant of the std matrix
        Args:
            std: either a diagonal, cholesky, or sqrt matrix depending on the policy
        Returns:
            The log determinant of std, aka log sum the diagonal
        """
        pass

    @abstractmethod
    def maha(self, mean, mean_other, std) -> ch.Tensor:
        """
        Compute the mahalanbis distance between two means. std is the scaling matrix.
        Args:
            mean: left mean
            mean_other: right mean
            std: scaling matrix

        Returns:
            Mahalanobis distance between mean and mean_other
        """
        pass

    @abstractmethod
    def precision(self, std: ch.Tensor) -> ch.Tensor:
        """
        Compute precision matrix given the std.
        Args:
            std: std matrix

        Returns:
            Precision matrix
        """
        pass

    @abstractmethod
    def covariance(self, std) -> ch.Tensor:
        """
        Compute the full covariance matrix given the std.
        Args:
            std: std matrix

        Returns:

        """
        pass

    @abstractmethod
    def set_std(self, std: ch.Tensor) -> None:
        """
        For the NON-contextual case we do not need to regress the std, we can simply set it. 
        This is a helper method to achieve this.
        Args:
            std: projected std

        Returns:

        """
        pass

    def get_last_layer(self):
        """
        Returns last layer of network. Only required for the PAPI projection.

        Returns:
            Last layer weights for PAPI prpojection. 

        """
        return self._affine_layers[-1].weight.data

    def papi_weight_update(self, eta: ch.Tensor, A: ch.Tensor):
        """
        Update the last layer alpha according to papi paper [Akrour et al., 2019]
        Args:
            eta: Multiplier alpha from [Akrour et al., 2019]
            A: Projected intermediate policy matrix

        Returns:

        """
        self._affine_layers[-1].weight.data *= eta
        self._affine_layers[-1].weight.data += (1 - eta) * A

    @property
    def is_root(self):
        """
        Whether policy is returning a full sqrt matrix as std.
        Returns:
        
        """
        return False

    @property
    def is_diag(self):
        """
        Whether the policy is returning a diagonal matrix as std.
        Returns:

        """
        return False
