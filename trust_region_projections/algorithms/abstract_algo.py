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

import abc
import logging
import torch as ch
from typing import Tuple, Union

from trust_region_projections.models.policy.abstract_gaussian_policy import AbstractGaussianPolicy
from trust_region_projections.projections.base_projection_layer import BaseProjectionLayer
from trust_region_projections.trajectories.trajectory_sampler import TrajectorySampler
from trust_region_projections.utils.custom_store import CustomStore


class AbstractAlgorithm(abc.ABC):
    def __init__(self,
                 policy: AbstractGaussianPolicy,
                 env_runner: TrajectorySampler,
                 projection: BaseProjectionLayer,

                 train_steps: int = 1000,
                 max_grad_norm: Union[float, None] = 0.5,

                 discount_factor: float = 0.99,

                 store: CustomStore = None,
                 advanced_logging: bool = True,
                 log_interval: int = 5,
                 save_interval: int = -1,

                 seed: int = 1,
                 cpu: bool = True,
                 dtype: ch.dtype = ch.float32):
        """
        Policy gradient that can be extended to PPO, PAPI, and to work with projections layers from [Otto, et al. 2021].
        Args:
            env_runner: Takes care of generating trajectory samples.
            policy: An `AbstractPolicy` which maps observations to action distributions.
            train_steps: Total number of training steps.
            max_grad_norm: Gradient norm clipping.
            discount_factor: Discount factor for return computation.
            store: Cox store
            advanced_logging: Add more logging output.
            log_interval: How often to log.
            save_interval: How often to save model.
            seed: Seed for generating envs
            cpu: Compute on CPU only.
            dtype: Data type to use, either of float32 or float64. The later might be necessary for higher
                    dimensions in order to learn the full covariance.
        """

        # Policy
        self.policy = policy

        # training steps
        self.train_steps = train_steps
        self.max_grad_norm = max_grad_norm
        self._global_steps = 0

        # Environments
        self.env_runner = env_runner

        # projection
        self.projection = projection

        # loss and reward parameters
        self.discount_factor = discount_factor

        # Config
        self.seed = seed
        self.cpu = cpu
        self.dtype = dtype

        # logging
        self.save_interval = save_interval
        self.advanced_logging = advanced_logging
        self.log_interval = log_interval

        self.store = store

        self._logger = logging.getLogger('abstract_algorithm')

    def setup_stores(self):
        """
        Setup hdf5 storages for saving training curves and metrics.

        Returns:
        """
        reward_schema = {
            'mean': float,
            'std': float,
            'min': float,
            'max': float,
            'step_reward': float,
            'length': float,
            'length_std': float,
        }
        self.store.add_table('exploration_reward', reward_schema)
        self.store.add_table('evaluation_reward', reward_schema)

        # Table for final results
        self.store.add_table('final_results', {
            'iteration': int,
            '5_rewards': float,
            '5_rewards_test': float
        })

        constraint_schema = {
            'kl': float,
            'constraint': float,
            'mean_constraint': float,
            'cov_constraint': float,
            'entropy': float,
            'entropy_diff': float,
            'kl_max': float,
            'constraint_max': float,
            'mean_constraint_max': float,
            'cov_constraint_max': float,
            'entropy_max': float,
            'entropy_diff_max': float,
        }

        self.store.add_table('constraints', constraint_schema)

        if self.advanced_logging:
            self.store.add_table('distribution', {
                'mean': self.store.PICKLE,
                'std': self.store.PICKLE,
            })

            if self.projection and self.projection.do_regression:
                self.store.add_table('constraints_initial', constraint_schema)
                self.store.add_table('constraints_projection', constraint_schema)

    def evaluate_policy(self, logging_step: int, render: bool = False, deterministic: bool = True,
                        generate_plots: bool = False):
        """
        Evaluates the current policy on the test environments.
        Args:
            logging_step: Current step for tb logging
            render: Render Policy (if applicable)
            deterministic: Make policy actions deterministic for testing
            generate_plots: Generate plots (if applicable)

        Returns:
            exploration_dict, evaluation_dict
        """
        exploration_dict = self.env_runner.get_exploration_performance()
        evaluation_dict = self.env_runner.evaluate_policy(self.policy, render=render, deterministic=deterministic)

        if self.log_interval != 0 and self._global_steps % self.log_interval == 0:
            if self.advanced_logging:
                self._logger.info(self.generate_reward_string(exploration_dict, "train"))
                self._logger.info(self.generate_reward_string(evaluation_dict))

            self.store.log_table_and_tb('exploration_reward', exploration_dict, step=logging_step)
            self.store['exploration_reward'].flush_row()

            self.store.log_table_and_tb('evaluation_reward', evaluation_dict, step=logging_step)
            self.store['evaluation_reward'].flush_row()

        return exploration_dict, evaluation_dict

    @staticmethod
    def generate_reward_string(reward_dict: dict, type="test"):
        """
        Generate string for printing agent performance.
        Args:
            reward_dict: Dict with metrics
            type: 'train' or 'test'
        Returns:
            String with relevant metrics for logging
        """
        return f"Avg. {type} reward: {reward_dict['mean']:.4f} +/- {reward_dict['std']:.4f}| " \
               f"Min/Max {type} reward: {reward_dict['min']:.4f}/{reward_dict['max']:.4f} | " \
               f"Avg. step {type} reward: {reward_dict['step_reward']:.4f} | " \
               f"Avg. {type} episode length: {reward_dict['length']:.4f} +/- " \
               f"{reward_dict['length_std'] :.2f}"

    def regression_step(self, obs: ch.Tensor, q: Tuple[ch.Tensor, ch.Tensor], n_minibatches: int, logging_step: int):
        """
        Execute additional regression steps to match policy output and projection.
        The policy parameters are updated in-place.
        Args:
            obs: Observations from trajectories
            q: Old distribution
            n_minibatches: Batch size for regression
            logging_step: Step index for logging

        Returns:
            dict of mean regression loss
        """
        if self.advanced_logging and self.projection.do_regression:
            # get prediction before the regression to compare to regressed policy
            with ch.no_grad():
                p = self.policy(obs)
                p_proj = self.projection(self.policy, p, q, self._global_steps)

            self.store.log_table_and_tb('constraints_initial',
                                        self.projection.compute_metrics(self.policy, p, q), step=logging_step)
            self.store.log_table_and_tb('constraints_projection',
                                        self.projection.compute_metrics(self.policy, p_proj, q), step=logging_step)
            self.store['constraints_initial'].flush_row()
            self.store['constraints_projection'].flush_row()

        return self.projection.trust_region_regression(self.policy, obs, q, n_minibatches, self._global_steps)

    def log_metrics(self, obs, q, logging_step):
        """
        Computes and logs the trust region metrics.
        Args:
            obs: Observations used for evaluation
            q: Old distributions
            logging_step: Current logging step

        Returns:
            Dict of trust region metrics
        """
        with ch.no_grad():
            p = self.policy(obs)
        metrics_dict = self.projection.compute_metrics(self.policy, p, q)

        if self.log_interval != 0 and self._global_steps % self.log_interval == 0:
            self.store.log_table_and_tb('constraints', metrics_dict, step=logging_step)
            self.store['constraints'].flush_row()

        return metrics_dict

    @abc.abstractmethod
    def step(self):
        pass

    @abc.abstractmethod
    def learn(self):
        pass

    @abc.abstractmethod
    def save(self, iteration):
        pass

    @staticmethod
    @abc.abstractmethod
    def agent_from_data(store, train_steps=None):
        pass

    @staticmethod
    @abc.abstractmethod
    def agent_from_params(params, store=None):
        pass
