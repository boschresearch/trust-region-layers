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

import numpy as np
import torch as ch
import torch.nn as nn
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from typing import Iterable, Sequence, Union


def initialize_weights(model: nn.Module, initialization_type: str, scale: float = 2 ** 0.5, init_w=3e-3):
    """
    Weight initializer for the layer or model.
    Args:
        model: module to initialize
        initialization_type: type of inialization
        scale: gain value for orthogonal init
        init_w: init weight for normal and uniform init
    Returns: 
    """

    for p in model.parameters():
        if initialization_type == "normal":
            if len(p.data.shape) >= 2:
                p.data.normal_(init_w)  # 0.01
            else:
                p.data.zero_()
        elif initialization_type == "uniform":
            if len(p.data.shape) >= 2:
                p.data.uniform_(-init_w, init_w)
            else:
                p.data.zero_()
        elif initialization_type == "xavier":
            if len(p.data.shape) >= 2:
                nn.init.xavier_normal_(p.data)
            else:
                p.data.zero_()
        elif initialization_type == "orthogonal":
            if len(p.data.shape) >= 2:
                nn.init.orthogonal_(p.data, gain=scale)
            else:
                p.data.zero_()
        else:
            raise ValueError(
                "Not a valid initialization type. Choose one of 'normal', 'uniform', 'xavier', and 'orthogonal'")


def get_mlp(input_dim: int, hidden_sizes: Sequence[int], kernel_init: str, use_bias: bool = True):
    """
    create the hidden part of an MLP
    Args:
        input_dim: size of previous layer/input
        hidden_sizes: iterable of hidden unit sizes
        kernel_init: kernel initializer
        use_bias: use bias of dense layer

    Returns:
        ModuleList of all layers

    """

    affine_layers = nn.ModuleList()

    prev = input_dim
    for l in hidden_sizes:
        x = nn.Linear(prev, l, bias=use_bias)
        initialize_weights(x, kernel_init)
        affine_layers.append(x)
        prev = l

    return affine_layers


def get_activation(activation_type: str, **kwargs):
    """
    Create torch activation function instance
    Args:
        activation_type:
    Returns:
         Torch activation function instance
    """
    if activation_type.lower() == "tanh":
        return nn.Tanh()
    elif activation_type.lower() == "relu":
        return nn.ReLU(**kwargs)
    elif activation_type.lower() == "leaky_relu":
        return nn.LeakyReLU(**kwargs)
    elif activation_type.lower() == "prelu":
        return nn.PReLU(**kwargs)
    elif activation_type.lower() == "celu":
        return nn.CELU(**kwargs)
    else:
        ValueError(f"Optimizer {activation_type} is not supported.")


def get_optimizer(optimizer_type: str, model_parameters: Union[Iterable[ch.Tensor], Iterable[dict]],
                  learning_rate: float, **kwargs):
    """
    Get optimizer instance for given model parameters
    Args:
        model_parameters:
        optimizer_type:
        learning_rate:
        **kwargs:

    Returns:
        torch optimizer
    """
    if optimizer_type.lower() == "sgd":
        return optim.SGD(model_parameters, learning_rate, **kwargs)
    elif optimizer_type.lower() == "adam":
        return optim.Adam(model_parameters, learning_rate, **kwargs)
    elif optimizer_type.lower() == "adamw":
        return optim.AdamW(model_parameters, learning_rate, **kwargs)
    elif optimizer_type.lower() == "adagrad":
        return optim.adagrad.Adagrad(model_parameters, learning_rate, **kwargs)
    else:
        ValueError(f"Optimizer {optimizer_type} is not supported.")


def get_lr_schedule(schedule_type: str, optimizer: Optimizer, total_iters: int) -> Union[
    None, optim.lr_scheduler.LambdaLR]:
    """
    Generate lr schedule instance
        schedule_type: What type of schedule to generate
        optimizer: Optimizer instance whose lr is controlled by the schedule.
        total_iters: total number of iterations. Required for linear decaying schedule.
    Returns:
        Lr schedule instance
    """
    if not schedule_type or schedule_type.isspace():
        return None

    elif schedule_type.lower() == "linear":
        # Set adam learning rate to 3e-4 * alpha, where alpha decays from 1 to 0 over training
        lam = lambda epoch: 1 - epoch / total_iters
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lam)

    elif schedule_type.lower() == "papi":
        # Multiply learning rate with 0.8 every time the backtracking fails
        lam = lambda n_calls: 0.8 ** n_calls
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lam)

    else:
        raise ValueError(
            f"Learning rate schedule {schedule_type} is not supported. Select one of [None, linear, papi].")
