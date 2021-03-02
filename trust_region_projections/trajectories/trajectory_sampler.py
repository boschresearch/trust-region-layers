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

import collections
import logging
import numpy as np
import torch as ch
from typing import Union

from trust_region_projections.models.policy.abstract_gaussian_policy import AbstractGaussianPolicy
from trust_region_projections.models.value.vf_net import VFNet
from trust_region_projections.trajectories.dataclass import TrajectoryOnPolicyRaw
from trust_region_projections.trajectories.normalized_env_wrapper import NormalizedEnvWrapper
from trust_region_projections.utils.torch_utils import get_numpy, tensorize, to_gpu

logger = logging.getLogger("env_runner")


class TrajectorySampler(object):
    def __init__(self, env_id: str, n_envs: int = 1, n_test_envs=1, max_episode_length=1000,
                 discount_factor: float = 0.99, norm_obs: Union[bool, None] = bool, clip_obs: Union[float, None] = 10.0,
                 norm_rewards: Union[bool, None] = True, clip_rewards: Union[float, None] = 10.0, cpu: bool = True,
                 dtype=ch.float32, seed: int = 1):

        """
        Instance that takes care of generating Trajectory samples.
        Args:
           env_id: ID of training env
           n_envs: Number of parallel envs to run for more efficient sampling.
           max_episode_length: Sets env dones flag to True after n steps. (only necessary if env does not have
                    a time limit).
           n_test_envs: Number of environments to use during testing of the current policy.
           discount_factor: Discount factor for optional reward normalization.
           norm_obs: If true, keeps moving mean and variance of observations and normalizes new observations.
           clip_obs: Clipping value for normalized observations.
           norm_rewards: If true, keeps moving variance of rewards and normalizes incoming rewards.
           clip_rewards: lipping value for normalized rewards.
           cpu: Compute on CPU only.
           dtype: Data type to use, either of float32 or float64. The later might be necessary for higher
                   dimensions in order to learn the full covariance.
           seed: Seed for generating envs

        Returns:

        """

        self.dtype = dtype
        self.cpu = cpu
        self.n_envs = n_envs
        self.n_test_envs = n_test_envs

        self.total_rewards = collections.deque(maxlen=100)
        self.total_steps = collections.deque(maxlen=100)

        self.envs = NormalizedEnvWrapper(env_id, n_envs, n_test_envs, max_episode_length=max_episode_length,
                                         gamma=discount_factor, norm_obs=norm_obs, clip_obs=clip_obs,
                                         norm_rewards=norm_rewards, clip_rewards=clip_rewards, seed=seed)

    def run(self, rollout_steps, policy: AbstractGaussianPolicy, vf_model: Union[VFNet, None] = None,
            reset_envs: bool = False) -> TrajectoryOnPolicyRaw:
        """
        Generate trajectories of the environment.
        Args:
            rollout_steps: Number of steps to generate
            policy: Policy model to generate samples for
            vf_model: vf model to generate value estimate for all states.
            reset_envs: Whether to reset all envs in the beginning.

        Returns:
            Trajectory with the respective data as torch tensors.
        """

        # Here, we init the lists that will contain the mb of experiences
        num_envs = self.n_envs

        base_shape = (rollout_steps, num_envs)
        base_shape_p1 = (rollout_steps + 1, num_envs)
        base_action_shape = base_shape + self.envs.action_space.shape

        mb_obs = ch.zeros(base_shape_p1 + self.envs.observation_space.shape, dtype=self.dtype)
        mb_actions = ch.zeros(base_action_shape, dtype=self.dtype)
        mb_rewards = ch.zeros(base_shape, dtype=self.dtype)
        mb_dones = ch.zeros(base_shape, dtype=ch.bool)
        ep_infos = []

        mb_time_limit_dones = ch.zeros(base_shape, dtype=ch.bool)
        mb_means = ch.zeros(base_action_shape, dtype=self.dtype)
        mb_stds = ch.zeros(base_action_shape + self.envs.action_space.shape, dtype=self.dtype)

        # continue from last state
        # Before first step we already have self.obs because env calls self.obs = env.reset() on init
        obs = self.envs.reset() if reset_envs else self.envs.last_obs
        obs = tensorize(obs, self.cpu, self.dtype)

        # For n in range number of steps
        for i in range(rollout_steps):
            # Given observations, get action value and lopacs
            pds = policy(obs, train=False)
            actions = policy.sample(pds)
            squashed_actions = policy.squash(actions)

            mb_obs[i] = obs
            mb_actions[i] = squashed_actions

            obs, rewards, dones, infos = self.envs.step(squashed_actions.cpu().numpy())
            obs = tensorize(obs, self.cpu, self.dtype)

            mb_means[i] = pds[0]
            mb_stds[i] = pds[1]
            mb_time_limit_dones[i] = tensorize(infos["horizon"], self.cpu, ch.bool)

            if infos.get("done"):
                ep_infos.extend(infos.get("done"))

            mb_rewards[i] = tensorize(rewards, self.cpu, self.dtype)
            mb_dones[i] = tensorize(dones, self.cpu, ch.bool)

        # need value prediction for last obs in rollout to estimate loss
        mb_obs[-1] = obs

        # compute all logpacs and value estimates at once --> less computation
        mb_logpacs = policy.log_probability((mb_means, mb_stds), mb_actions)
        mb_values = (vf_model if vf_model else policy.get_value)(mb_obs, train=False)

        out = (mb_obs[:-1], mb_actions, mb_logpacs, mb_rewards, mb_values,
               mb_dones, mb_time_limit_dones, mb_means, mb_stds)

        if not self.cpu:
            out = tuple(map(to_gpu, out))

        if ep_infos:
            ep_infos = np.array(ep_infos)
            ep_length, ep_reward = ep_infos[:, 0], ep_infos[:, 1]
            self.total_rewards.extend(ep_reward)
            self.total_steps.extend(ep_length)

        return TrajectoryOnPolicyRaw(*out)

    def evaluate_policy(self, policy: AbstractGaussianPolicy, render: bool = False, deterministic: bool = True):
        """
        Evaluate a given policy
        Args:
            policy: policy to evaluate
            render: render policy behavior
            deterministic: choosing deterministic actions

        Returns:
            Dict with performance metrics.
        """
        if self.n_test_envs == 0:
            return
        n_runs = 1
        ep_rewards = np.zeros((n_runs, self.n_test_envs,))
        ep_lengths = np.zeros((n_runs, self.n_test_envs,))

        for i in range(n_runs):
            not_dones = np.ones((self.n_test_envs,), np.bool)
            obs = self.envs.reset_test()
            while np.any(not_dones):
                ep_lengths[i, not_dones] += 1
                if render:
                    self.envs.render_test(mode="human")
                with ch.no_grad():
                    p = policy(tensorize(obs, self.cpu, self.dtype))
                    actions = p[0] if deterministic else policy.sample(p)
                    actions = policy.squash(actions)
                obs, rews, dones, infos = self.envs.step_test(get_numpy(actions))
                ep_rewards[i, not_dones] += rews[not_dones]

                # only set to False when env has never terminated before, otherwise we favor earlier terminating envs.
                not_dones = np.logical_and(~dones, not_dones)

        return self.get_reward_dict(ep_rewards, ep_lengths)

    def get_exploration_performance(self):
        ep_reward = np.array(self.total_rewards)
        ep_length = np.array(self.total_steps)
        return self.get_reward_dict(ep_reward, ep_length)

    @staticmethod
    def get_reward_dict(ep_reward, ep_length):
        return {
            'mean': ep_reward.mean().item(),
            'std': ep_reward.std().item(),
            'max': ep_reward.max().item(),
            'min': ep_reward.min().item(),
            'step_reward': (ep_reward / ep_length).mean().item(),
            'length': ep_length.mean().item(),
            'length_std': ep_length.std().item(),
        }

    @property
    def observation_space(self):
        return self.envs.observation_space

    @property
    def observation_shape(self):
        return self.observation_space.shape

    @property
    def action_space(self):
        return self.envs.action_space

    @property
    def action_shape(self):
        return self.action_space.shape
