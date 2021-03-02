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

import gym
from typing import Union

from trust_region_projections.trajectories.env_normalizer import BaseNormalizer, MovingAvgNormalizer
from trust_region_projections.trajectories.vector_env import SequentialVectorEnv


def make_env(env_id: str, seed: int, rank: int) -> callable:
    """
    Returns callable to create gym environment

    Args:
        env_id: gym env ID
        seed: seed for env
        rank: rank if multiple ensv are used

    Returns: callable for env constructor

    """

    def _get_env():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env

    return _get_env


class NormalizedEnvWrapper(object):

    def __init__(self, env_id: str, n_envs: int = 1, n_test_envs: int = 1, max_episode_length: int = 1000, gamma=0.99,
                 norm_obs: Union[bool, None] = True, clip_obs: Union[float, None] = None,
                 norm_rewards: Union[bool, None] = True, clip_rewards: Union[float, None] = None, seed: int = 1):
        """
        A vectorized gym environment wrapper that normalizes observations and returns.
        Args:
           env_id: ID of training env
           n_envs: Number of parallel envs to run for more efficient sampling.
           n_test_envs: Number of environments to use during testing of the current policy.
           max_episode_length: Sets env dones flag to True after n steps. (only necessary if env does not have
                    a time limit).
           gamma: Discount factor for optional reward normalization.
           norm_obs: If true, keeps moving mean and variance of observations and normalizes new observations.
           clip_obs: Clipping value for normalized observations.
           norm_rewards: If true, keeps moving variance of rewards and normalizes incoming rewards.
           clip_rewards: lipping value for normalized rewards.
           seed: Seed for generating envs
        """

        self.max_episode_length = max_episode_length

        self.envs = SequentialVectorEnv([make_env(env_id, seed, i) for i in range(n_envs)],
                                        max_episode_length=max_episode_length)
        if n_test_envs:
            # Create test envs here to leverage the moving average normalization for testing envs.
            self.envs_test = SequentialVectorEnv([make_env(env_id, seed + n_envs, i) for i in range(n_test_envs)],
                                                 max_episode_length=max_episode_length)

        self.norm_obs = norm_obs
        self.clip_obs = clip_obs
        self.norm_rewards = norm_rewards
        self.clip_rewards = clip_rewards

        ################################################################################################################
        # Support for state normalization or using time as a feature

        self.state_normalizer = BaseNormalizer()
        if self.norm_obs:
            # set gamma to 0 because we do not want to normalize based on return trajectory
            self.state_normalizer = MovingAvgNormalizer(self.state_normalizer, shape=self.observation_space.shape,
                                                        center=True, scale=True, gamma=0., clip=clip_obs)
        ################################################################################################################
        # Support for return normalization
        self.reward_normalizer = BaseNormalizer()
        if self.norm_rewards:
            self.reward_normalizer = MovingAvgNormalizer(self.reward_normalizer, shape=(), center=False, scale=True,
                                                         gamma=gamma, clip=clip_rewards)

        ################################################################################################################

        # save last of in env to return later to
        self.last_obs = self.envs.reset()

    def step(self, actions):

        obs, rews, dones, infos = self.envs.step(actions)

        self.last_obs = self.state_normalizer(obs)
        rews_norm = self.reward_normalizer(rews)

        self.state_normalizer.reset(dones)
        self.reward_normalizer.reset(dones)

        return self.last_obs.copy(), rews_norm, dones, infos

    def step_test(self, action):

        obs, rews, dones, infos = self.envs_test.step(action)

        obs = self.state_normalizer(obs, update=False)

        # Return unnormalized rewards for testing to assess performance
        return obs, rews, dones, infos

    def reset_test(self):
        obs = self.envs_test.reset()
        return self.state_normalizer(obs, update=False)

    def reset(self):

        self.state_normalizer.reset()
        self.reward_normalizer.reset()

        obs = self.envs.reset()
        return self.state_normalizer(obs)

    def render_test(self, mode="human", **kwargs):
        self.envs_test.render(mode, **kwargs)

    @property
    def observation_space(self):
        return self.envs.observation_space

    @property
    def action_space(self):
        return self.envs.action_space
