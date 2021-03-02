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

from collections import defaultdict
from typing import Sequence

import gym
import numpy as np


class SequentialVectorEnv(gym.Env):

    def __init__(self, env_fns: Sequence[callable], max_episode_length=np.inf):
        """
        Sequential vector env, that used multiple gym environments and executes the actions sequentially.
        Args:
            env_fns: Sequence of callables to create environments
            max_episode_length: Sets env dones flag to True after n steps. (only necessary if env does not have
                                a time limit).
        """

        self.envs = [f() for f in env_fns]
        self.num_envs = len(env_fns)
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

        self.max_episode_length = max_episode_length
        self.length_counter = np.zeros((self.num_envs,))
        self.total_ep_reward = np.zeros((self.num_envs,))

    def step(self, actions):
        """

        Simulate a "step" by several actors on their respective environments
        Inputs:
        - actions, list of actions to take
        - envs, list of the environments in which to take the actions
        Returns:
        - ep_info, a variable-length list of final rewards and episode lengths
            for the actors which have completed
        - rewards, a actors-length tensor with the rewards collected
        - states, a (actors, ... state_shape) tensor with resulting states
        - dones, an actors-length tensor with 1 if terminal, 0 otw
        """
        rewards, dones = np.zeros(self.num_envs), np.zeros(self.num_envs, np.bool)
        states = np.zeros((self.num_envs,) + self.observation_space.shape)
        ep_info = defaultdict(list)

        for i, (action, env) in enumerate(zip(actions, self.envs)):
            obs, rew, done, info = env.step(action)

            self.length_counter[i] += 1
            self.total_ep_reward[i] += rew

            max_horizon = self.length_counter[i] == self.max_episode_length
            done = done or max_horizon

            ep_info["horizon"].append(max_horizon)

            if done:
                obs = env.reset()

            if done:
                # return stats after max episode length in order to evaluate the exploration policy performance
                ep_info["done"].append((self.length_counter[i], self.total_ep_reward[i]))
                self.length_counter[i] = 0.
                self.total_ep_reward[i] = 0.

            # Aggregate
            ep_info["info"].append(info)
            states[i] = obs
            rewards[i] = rew
            dones[i] = done

        return np.vstack(states), np.array(rewards), np.array(dones), ep_info

    def reset(self):
        return np.vstack([env.reset() for env in self.envs])

    def render(self, mode='human'):
        if mode == "human":
            return self.envs[0].render()
        else:
            return [env.render(mode=mode) for env in self.envs]
