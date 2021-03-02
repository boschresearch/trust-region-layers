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
#

import logging
import math
from collections import deque
from typing import Union

import gym
import numpy as np
import torch as ch

from trust_region_projections.algorithms.abstract_algo import AbstractAlgorithm
from trust_region_projections.models.policy.abstract_gaussian_policy import AbstractGaussianPolicy
from trust_region_projections.models.policy.policy_factory import get_policy_network
from trust_region_projections.models.value.vf_net import VFNet
from trust_region_projections.projections.base_projection_layer import BaseProjectionLayer
from trust_region_projections.projections.projection_factory import get_projection_layer
from trust_region_projections.trajectories.dataclass import TrajectoryOnPolicy
from trust_region_projections.trajectories.trajectory_sampler import TrajectorySampler
from trust_region_projections.utils.custom_store import CustomStore
from trust_region_projections.utils.network_utils import get_lr_schedule, get_optimizer
from trust_region_projections.utils.torch_utils import flatten_batch, generate_minibatches, get_numpy, \
    select_batch, tensorize

logging.basicConfig(level=logging.INFO)


class PolicyGradient(AbstractAlgorithm):
    def __init__(self,
                 env_runner: TrajectorySampler,
                 policy: AbstractGaussianPolicy,
                 vf_model: VFNet,

                 optimizer_type: str = "adam",
                 optimizer_type_val: str = None,
                 learning_rate: float = 3e-4,
                 learning_rate_vf: float = None,

                 projection: BaseProjectionLayer = None,

                 train_steps: int = 1000,
                 epochs: int = 10,
                 val_epochs: int = 10,
                 n_minibatches: int = 4,

                 lr_schedule: str = "",
                 max_grad_norm: Union[float, None] = 0.5,

                 max_entropy_coeff: float = 0.0,
                 vf_coeff: float = 0.5,
                 entropy_penalty_coeff: float = 0.0,

                 rollout_steps: int = 2048,
                 discount_factor: float = 0.99,
                 use_gae: bool = True,
                 gae_scaling: float = 0.95,

                 norm_advantages: Union[bool, None] = True,
                 clip_advantages: Union[float, None] = None,

                 importance_ratio_clip: Union[float, None] = 0.2,
                 clip_vf: Union[float, None] = 0.2,

                 store: CustomStore = None,
                 advanced_logging: bool = True,
                 log_interval: int = 5,
                 save_interval: int = -1,

                 seed: int = 1,
                 cpu: bool = True,
                 dtype: ch.dtype = ch.float32
                 ):
        """
        Policy gradient that can be extended to PPO, PAPI, and to work with projections layers from [Otto, et al. 2021].
        Args:
            env_runner: Takes care of generating trajectory samples.
            policy: An `AbstractPolicy` which maps observations to action distributions.
                    Normally ConditionalGaussianPolicy is used.
            vf_model: An `AbstractPolicy` which returns the value prediction for input states.
                    Normally ConditionalGaussianPolicy is used.
            optimizer_type: Optimizer to use for the agent and vf.
            optimizer_type_val: Different vf optimizer if training separately.
            learning_rate: Learning rate for actor or joint optimizer.
            learning_rate_vf: Learning rate for (optional) vf optimizer.
            train_steps: Total number of training steps.
            epochs: Number of policy updates for each batch of sampled trajectories.
            val_epochs: Number of vf updates for each batch of sampled trajectories.
            n_minibatches: Number of minibatches for each batch of sampled trajectories.
            lr_schedule: Learning rate schedule type: 'linear' or ''
            max_grad_norm: Gradient norm clipping.
            max_entropy_coeff: Coefficient when complementing the reward with a max entropy objective.
            vf_coeff: Multiplier for vf loss to balance with policy gradient loss.
                    Default to `0.0` trains vf and policy separately.
                    `0.5` , which was used by OpenAI trains jointly.
            entropy_penalty_coeff: Coefficient for entropy regularization loss term.

            rollout_steps: Number of rollouts per environment (Batch size is rollout_steps * n_envs)
            discount_factor: Discount factor for return computation.
            use_gae: Use generalized advantage estimation for computing per-timestep advantage.
            gae_scaling: Lambda parameter for TD-lambda return and GAE.
            norm_advantages: If `True`, standardizes advantages for each update.
            clip_advantages: Value above and below to clip normalized advantages.
            importance_ratio_clip: Epsilon in clipped, surrogate PPO objective.
            clip_vf: Difference between new and old value predictions are clipped to this threshold.
            store: Cox store
            advanced_logging: Add more logging output.
            log_interval: How often to log.
            save_interval: How often to save model.
            seed: Seed for generating envs
            cpu: Compute on CPU only.
            dtype: Data type to use, either of float32 or float64. The later might be necessary for higher
                    dimensions in order to learn the full covariance.
        """

        super().__init__(policy, env_runner, projection, train_steps, max_grad_norm, discount_factor, store,
                         advanced_logging, log_interval, save_interval, seed, cpu, dtype)

        # training
        self.epochs = epochs
        self.val_epochs = val_epochs
        self.n_minibatches = n_minibatches

        # normalizing and clipping
        self.norm_advantages = norm_advantages
        self.clip_advantages = clip_advantages or 0.0
        self.clip_vf = clip_vf or 0.0
        self.importance_ratio_clip = importance_ratio_clip or 0.0

        # GAE
        self.use_gae = use_gae
        self.gae_scaling = gae_scaling or 0.0

        # loss parameters
        self.max_entropy_coeff = max_entropy_coeff
        self.vf_coeff = vf_coeff or 0.0
        self.entropy_coeff = entropy_penalty_coeff or 0.0

        # environment
        self.rollout_steps = rollout_steps

        # vf model
        self.vf_model = vf_model

        # optimizer
        self.optimizer = get_optimizer(optimizer_type, self.policy.parameters(), learning_rate, eps=1e-5)
        self.lr_schedule = get_lr_schedule(lr_schedule, self.optimizer, self.train_steps)
        if vf_model:
            self.optimizer_vf = get_optimizer(optimizer_type_val, self.vf_model.parameters(), learning_rate_vf,
                                              eps=1e-5)
            self.lr_schedule_vf = get_lr_schedule(lr_schedule, self.optimizer_vf, self.train_steps)

        if self.store:
            self.setup_stores()

        self.logger = logging.getLogger('policy_gradient')

    def setup_stores(self):
        # Logging setup
        super(PolicyGradient, self).setup_stores()

        loss_dict = {
            'loss': float,
            'vf_loss': float,
            'policy_loss': float,
            'entropy_loss': float,
            'trust_region_loss': float,
        }

        if self.projection.do_regression:
            loss_dict.update({'regression_loss': float})

        self.store.add_table('loss', loss_dict)

        if self.lr_schedule:
            lr_dict = {}
            lr_dict.update({f"lr": float})

            if self.lr_schedule_vf:
                lr_dict.update({f"lr_vf": float})
            self.store.add_table('lr', lr_dict)

    def advantage_and_return(self, rewards: ch.Tensor, values: ch.Tensor, dones: ch.Tensor,
                             time_limit_dones: ch.Tensor):
        """
        Calculate advantage (with GAE) and discounted returns.
        Further, provides specific treatment for terminal states which reached an artificial maximum horizon.

        GAE: h_t^V = r_t + y * V(s_t+1) - V(s_t)
        with
        V(s_t+1) = {0 if s_t is terminal
                   {V(s_t+1) if s_t not terminal and t != T (last step)
                   {V(s) if s_t not terminal and t == T

        Args:
            rewards: Rewards from environment
            values: Value estimates
            dones: Done flags for true termination
            time_limit_dones: Done flags for reaching artificial maximum time limit

        Returns:
            Advantages and Returns

        """
        returns = tensorize(np.zeros((self.rollout_steps + 1, self.env_runner.n_envs)), self.cpu, self.dtype)
        masks = ~dones
        time_limit_masks = ~time_limit_dones

        if self.use_gae:
            gae = 0
            for step in reversed(range(rewards.size(0))):
                delta = rewards[step] + self.discount_factor * values[step + 1] * masks[step] - values[step]
                gae = delta + self.discount_factor * self.gae_scaling * masks[step] * gae
                # when artificial time limit is reached, take current state's vf estimate as V(s) = r + yV(s')
                gae = gae * time_limit_masks[step]
                returns[step] = gae + values[step]
        else:
            returns[-1] = values[-1]
            for step in reversed(range(rewards.size(0))):
                returns[step] = (returns[step + 1] * self.discount_factor * masks[step] + rewards[step]) * \
                                time_limit_masks[step] + time_limit_dones[step] * values[step]

        returns = returns[:-1]
        advantages = returns - values[:-1]

        return advantages.clone().detach(), returns.clone().detach()

    def surrogate_loss(self, advantages: ch.Tensor, new_logpacs: ch.Tensor, old_logpacs: ch.Tensor):
        """
        Computes the surrogate reward for IS policy gradient R(\theta) = E[r_t * A_t]
        Optionally, clamping the ratio (for PPO) R(\theta) = E[clamp(r_t, 1-e, 1+e) * A_t]
        Args:
            advantages: unnormalized advantages
            new_logpacs: Log probabilities from current policy
            old_logpacs: Log probabilities
        Returns:
            The surrogate loss as described above
        """

        # Normalized Advantages
        if self.norm_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        if self.clip_advantages > 0:
            advantages = ch.clamp(advantages, -self.clip_advantages, self.clip_advantages)

        # Ratio of new probabilities to old ones
        ratio = (new_logpacs - old_logpacs).exp()

        surrogate_loss = ratio * advantages

        # PPO clipped ratio
        if self.importance_ratio_clip > 0:
            ratio_clipped = ratio.clamp(1 - self.importance_ratio_clip, 1 + self.importance_ratio_clip)
            surrogate_loss2 = ratio_clipped * advantages
            surrogate_loss = ch.min(surrogate_loss, surrogate_loss2)

        return -surrogate_loss.mean()

    def value_loss(self, values: ch.Tensor, returns: ch.Tensor, old_vs: ch.Tensor):
        """
        Computes the value function loss.

        When using GAE we have L_t = ((v_t + A_t).detach() - v_{t})
        Without GAE we get L_t = (r(s,a) + y*V(s_t+1) - v_{t}) accordingly.

        Optionally, we clip the value function around the original value of v_t

        Returns:
        Args:
            values: Current value estimates
            returns: Computed returns with GAE or n-step
            old_vs: Old value function estimates from behavior policy

        Returns:
            Value function loss as described above.
        """

        vf_loss = (returns - values).pow(2)

        if self.clip_vf > 0:
            # In OpenAI's PPO implementation, we clip the value function around the previous value estimate
            # and use the worse of the clipped and unclipped versions to train the value function
            vs_clipped = old_vs + (values - old_vs).clamp(-self.clip_vf, self.clip_vf)
            vf_loss_clipped = (vs_clipped - returns).pow(2)
            vf_loss = ch.max(vf_loss, vf_loss_clipped)

        return vf_loss.mean()

    def policy_step(self, dataset: TrajectoryOnPolicy):
        """
        Policy optimization step
        Args:
            dataset: NameTuple with obs, actions, logpacs, returns, advantages, values, and q
        Returns:
            Dict with total loss, policy loss, entropy loss, and trust region loss
        """

        obs, actions, old_logpacs, returns, advantages, q = \
            dataset.obs, dataset.actions, dataset.logpacs, dataset.returns, dataset.advantages, dataset.q

        losses, vf_losses, surrogates, entropy_losses, trust_region_losses = \
            [tensorize(0., self.cpu, self.dtype) for _ in range(5)]

        # set initial entropy value in first step to calculate appropriate entropy decay
        if self.projection.initial_entropy is None:
            self.projection.initial_entropy = self.policy.entropy(q).mean()

        for _ in range(self.epochs):
            batch_indices = generate_minibatches(obs.shape[0], self.n_minibatches)

            # Minibatches SGD
            for indices in batch_indices:
                batch = select_batch(indices, obs, actions, old_logpacs, advantages, q[0], q[1])
                b_obs, b_actions, b_old_logpacs, b_advantages, b_old_mean, b_old_std = batch
                b_q = (b_old_mean, b_old_std)

                p = self.policy(b_obs)
                proj_p = self.projection(self.policy, p, b_q, self._global_steps)

                new_logpacs = self.policy.log_probability(proj_p, b_actions)

                # Calculate policy rewards
                surrogate_loss = self.surrogate_loss(b_advantages, new_logpacs, b_old_logpacs)

                # Calculate entropy bonus
                entropy_loss = -self.entropy_coeff * self.policy.entropy(proj_p).mean()

                # Trust region loss
                trust_region_loss = self.projection.get_trust_region_loss(self.policy, p, proj_p)

                # Total loss
                loss = surrogate_loss + entropy_loss + trust_region_loss

                # If we are sharing weights, take the value step simultaneously
                if self.vf_coeff > 0 and not self.vf_model:
                    # if no vf model is present, the model is part of the policy, therefore has to be trained jointly
                    batch_vf = select_batch(indices, returns, dataset.values)
                    vs = self.policy.get_value(b_obs)
                    vf_loss = self.value_loss(vs, *batch_vf)  # b_returns, b_old_values)
                    loss += self.vf_coeff * vf_loss
                    vf_losses += vf_loss.detach()

                self.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm > 0:
                    ch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                surrogates += surrogate_loss.detach()
                trust_region_losses += trust_region_loss.detach()
                entropy_losses += entropy_loss.detach()
                losses += loss.detach()

        steps = self.epochs * (math.ceil(obs.shape[0] / self.n_minibatches))
        loss_dict = {"loss": (losses / steps).detach(),
                     "policy_loss": (surrogates / steps).detach(),
                     "entropy_loss": (entropy_losses / steps).detach(),
                     "trust_region_loss": (trust_region_losses / steps).detach()
                     }

        if not self.vf_model:
            loss_dict.update({"vf_loss": (vf_losses / steps).detach()})

        if not self.policy.contextual_std and self.projection.proj_type not in ["ppo", "papi"]:
            # set policy with projection value without doing regression.
            # In non-contextual cases we have only one cov, so the projection is the same for all samples.
            self.policy.set_std(proj_p[1][0].detach())

        return loss_dict

    def value_step(self, dataset: TrajectoryOnPolicy):
        """
        Take an optimizer step fitting the value function
        Args:
            dataset: NameTuple with obs, returns, and values
        Returns:
            Dict with loss of the value regression

        """

        obs, returns, old_values = dataset.obs, dataset.returns, dataset.values

        vf_losses = tensorize(0., self.cpu, self.dtype)

        for _ in range(self.val_epochs):
            splits = generate_minibatches(obs.shape[0], self.n_minibatches)

            # Minibatch SGD
            for indices in splits:
                batch = select_batch(indices, returns, old_values, obs)

                sel_returns, sel_old_values, sel_obs = batch
                vs = self.vf_model(sel_obs)

                vf_loss = self.value_loss(vs, sel_returns, sel_old_values)

                self.optimizer_vf.zero_grad()
                vf_loss.backward()
                self.optimizer_vf.step()
                vf_losses += vf_loss.detach()

        steps = self.val_epochs * (math.ceil(obs.shape[0] / self.n_minibatches))
        return {"vf_loss": (vf_losses / steps)}

    def sample(self) -> TrajectoryOnPolicy:
        """
        Generate trajectory samples.
        Returns:
            NamedTuple with samples
        """
        with ch.no_grad():
            dataset = self.env_runner.run(self.rollout_steps, self.policy, self.vf_model)
            (obs, actions, logpacs, rewards, values, dones, time_limit_dones, old_means, old_stds) = dataset

            if self.max_entropy_coeff > 0:
                # add entropy to rewards in order to maximize trajectory of discounted reward + entropy
                # R = E[sum(y^t (r_t + a*H_t)]
                rewards += self.max_entropy_coeff * self.policy.entropy((old_means, old_stds)).detach()

            # Calculate advantages and returns
            advantages, returns = self.advantage_and_return(rewards, values, dones, time_limit_dones)

        # Unrolled trajectories (T, n_envs, ...) -> (T*n_envs, ...) to train in one forward pass
        unrolled = map(flatten_batch,
                       (obs, actions, logpacs, rewards, returns, advantages, values[:-1], dones, time_limit_dones))
        q_unrolled = tuple(map(flatten_batch, (old_means, old_stds)))

        return TrajectoryOnPolicy(*unrolled, q_unrolled)

    def step(self):
        """
        Take a full training step, including sampling, policy, and vf update.
        Returns:
            Dict with metrics, dict with train/test reward

        """
        self._global_steps += 1

        loss_dict = {}
        dataset = self.sample()

        if self.vf_model:
            # Train value network separately
            loss_dict.update(self.value_step(dataset))

        # Policy optimization step or in case the network shares weights/is trained jointly also value update
        loss_dict.update(self.policy_step(dataset))

        # PAPI projection after the policy updates with PPO.
        if self.projection.proj_type == "papi":
            self.projection(self.policy, None, dataset.q, self._global_steps,
                            obs=dataset.obs, lr_schedule=self.lr_schedule, lr_schedule_vf=self.lr_schedule_vf)

        self.lr_schedule_step()

        logging_step = self._global_steps * self.rollout_steps
        loss_dict.update(self.regression_step(dataset.obs, dataset.q, self.n_minibatches, logging_step))

        if self.log_interval != 0 and self._global_steps % self.log_interval == 0:
            self.store.log_table_and_tb('loss', loss_dict, step=logging_step)
            self.store['loss'].flush_row()

        metrics_dict = self.log_metrics(dataset.obs, dataset.q, logging_step)
        loss_dict.update(metrics_dict)

        exploration_dict, evaluation_dict = self.evaluate_policy(logging_step)

        return loss_dict, {"exploration": exploration_dict, "evaluation": evaluation_dict}

    def lr_schedule_step(self):
        """
        Update learning rates based on schedules
        Returns:
        """
        if self.lr_schedule:
            lr_dict = {}
            # Linear learning rate annealing
            # PAPI uses a different concept for lr decay that is implemented in its projection
            if not self.projection.proj_type == "papi": self.lr_schedule.step()
            lr_dict.update({f"lr": self.lr_schedule.get_last_lr()[0]})
            if self.lr_schedule_vf:
                if not self.projection.proj_type == "papi": self.lr_schedule_vf.step()
                lr_dict.update({f"lr_vf": self.lr_schedule_vf.get_last_lr()[0]})

            self.store.log_table_and_tb('lr', lr_dict, step=self._global_steps * self.rollout_steps)
            self.store['lr'].flush_row()

    def learn(self):
        """
        Train agent fully for specified number of train steps
        Returns:
            Dict with train/test reward metrics

        """
        rewards = deque(maxlen=5)
        rewards_test = deque(maxlen=5)

        epoch = self._global_steps
        for epoch in range(self._global_steps, self.train_steps):
            metrics_dict, rewards_dict = self.step()

            if self.log_interval != 0 and self._global_steps % self.log_interval == 0 and self.advanced_logging:
                self.logger.info("-" * 80)
                metrics = ", ".join((*map(lambda kv: f'{kv[0]}={get_numpy(kv[1]):.4f}', metrics_dict.items()),))
                self.logger.info(f"iter {epoch:6d}: {metrics}")

            if self.save_interval > 0 and epoch % self.save_interval == 0:
                self.save(epoch)

            rewards.append(rewards_dict['exploration']['mean'])
            rewards_test.append(rewards_dict['evaluation']['mean'])

        self.store["final_results"].append_row({
            'iteration': epoch,
            '5_rewards': np.array(rewards).mean(),
            '5_rewards_test': np.array(rewards).mean(),
        })

        # final evaluation and save of model
        if self.save_interval > 0:
            self.save(self.train_steps)
        exploration_dict, evaluation_dict = self.evaluate_policy(self._global_steps * self.rollout_steps)

        return {"exploration": exploration_dict, "evaluation": evaluation_dict}

    def save(self, iteration: int):
        """
        Create checkpoint of current training progress
        Args:
            iteration: Current iteration

        Returns:

        """
        checkpoint_dict = {
            'iteration': iteration,
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'env_runner': self.env_runner
        }

        if self.vf_model:
            checkpoint_dict.update({'vf_model': self.vf_model.state_dict(),
                                    'optimizer_vf': self.optimizer_vf.state_dict()
                                    })
        self.store['checkpoints'].append_row(checkpoint_dict)

    @staticmethod
    def agent_from_data(store: CustomStore, train_steps: Union[None, int] = None):
        """
        Initializes an agent from serialized data (via cox)
        Args:
            store: The cox store to load
            train_steps: Which checkpoint to load

        Returns:
            PG agent, dict with agent params
        """

        param_keys = list(store['metadata'].df.columns)

        def process_item(v):
            try:
                return v.item()
            except (ValueError, AttributeError):
                return v

        param_values = [process_item(store.load('metadata', v, "object")) for v in param_keys]
        agent_params = {k: v for k, v in zip(param_keys, param_values)}
        if train_steps is not None:
            agent_params['train_steps'] = train_steps
        agent = PolicyGradient.agent_from_params(agent_params)

        mapper = ch.device('cuda:0') if not agent_params['cpu'] else ch.device('cpu')
        iteration = store.load('checkpoints', 'iteration', '')

        def load_state_dict(model, ckpt_name):
            state_dict = store.load('checkpoints', ckpt_name, "state_dict", map_location=mapper)
            model.load_state_dict(state_dict)

        load_state_dict(agent.policy, 'policy')
        load_state_dict(agent.optimizer, 'optimizer')
        if agent.lr_schedule:
            agent.lr_schedule.last_epoch = iteration

        if agent.vf_model:
            load_state_dict(agent.vf_model, 'vf_model')
            # load_state_dict(agent.optimizer_vf, 'optimizer_vf')
            if agent.lr_schedule:
                agent.lr_schedule_vf.last_epoch = iteration

        agent.env_runner = store.load('checkpoints', 'env_runner', 'pickle')
        agent._global_steps = iteration + 1
        agent.store = store

        return agent, agent_params

    @staticmethod
    def agent_from_params(params: dict, store: Union[None, CustomStore] = None):
        """
        Construct a run given a dict of HPs.
        Args:
            params: param dict
            store: Cox logging instance.

        Returns:
            agent

        """

        print(params)

        env = gym.make(params['game'])
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        init = params['initialization']
        activation = params['activation']
        share_weights = params['share_weights']
        policy_type = params['policy_type']

        use_gpu = not params['cpu']
        device = ch.device("cuda:0" if use_gpu else "cpu")
        seed = params['seed']

        np.random.seed(seed)
        ch.manual_seed(seed)

        dtype = ch.float64 if params['dtype'] == "float64" else ch.float32

        # vf network
        vf_model = None
        if not share_weights:
            vf_model = VFNet(obs_dim, 1, init, hidden_sizes=params['hidden_sizes_vf'], activation=activation)
            vf_model = vf_model.to(device, dtype)

        # policy network
        policy = get_policy_network(policy_type, params['proj_type'], device=device, dtype=dtype,
                                    obs_dim=obs_dim, action_dim=action_dim, init=init,
                                    hidden_sizes=params['hidden_sizes_policy'], activation=activation,
                                    contextual_std=params['contextual_std'], init_std=params['init_std'],
                                    share_weights=share_weights, minimal_std=params['minimal_std'],
                                    vf_model=vf_model if params["vf_coeff"] != 0 else None)

        # environments
        env_runner = TrajectorySampler(params['game'], n_envs=params['n_envs'], n_test_envs=params['n_test_envs'],
                                       max_episode_length=params['max_episode_length'],
                                       discount_factor=params['discount_factor'], norm_obs=params['norm_observations'],
                                       clip_obs=params['clip_observations'] or 0.0, norm_rewards=params['norm_rewards'],
                                       clip_rewards=params['clip_rewards'] or 0.0, cpu=not use_gpu, dtype=dtype,
                                       seed=seed)

        # projections
        projection = get_projection_layer(
            proj_type=params['proj_type'],
            mean_bound=params['mean_bound'],
            cov_bound=params['cov_bound'],
            trust_region_coeff=params['trust_region_coeff'],
            scale_prec=params['scale_prec'],

            entropy_schedule=params['entropy_schedule'],
            action_dim=action_dim,
            total_train_steps=params['train_steps'],
            target_entropy=params['target_entropy'],
            temperature=params['temperature'],
            entropy_eq=params['entropy_eq'],
            entropy_first=params['entropy_first'],

            do_regression=params['do_regression'],
            regression_iters=params['regression_iters'],
            regression_lr=params['lr_reg'],
            optimizer_type_reg=params['optimizer_reg'],

            cpu=not use_gpu,
            dtype=dtype
        )

        advanced_logging = params['advanced_logging'] and store is not None
        log_interval = params['log_interval'] if store is not None else 0

        # Remove if interested in running with multiple CPUs for one job
        ch.set_num_threads(1)

        p = PolicyGradient(
            env_runner=env_runner,
            policy=policy,
            # only pass the model if trained jointly, otherwise, the vf is accessed through the policy.
            vf_model=vf_model if params["vf_coeff"] == 0. else None,

            optimizer_type=params['optimizer'],
            optimizer_type_val=params['optimizer_vf'],
            learning_rate=params['lr'],
            learning_rate_vf=params['lr_vf'],

            projection=projection,

            train_steps=params['train_steps'],
            epochs=params['epochs'],
            val_epochs=params['val_epochs'],
            n_minibatches=params['num_minibatches'],

            lr_schedule=params['lr_schedule'],
            max_grad_norm=params['clip_grad_norm'],

            max_entropy_coeff=params['max_entropy_coeff'],
            vf_coeff=params['vf_coeff'],
            entropy_penalty_coeff=params['entropy_penalty_coeff'],

            rollout_steps=params['rollout_steps'],
            discount_factor=params['discount_factor'],
            use_gae=params['use_gae'],
            gae_scaling=params['gae_scaling'],

            norm_advantages=params['norm_advantages'],
            clip_advantages=params['clip_advantages'],
            importance_ratio_clip=params['importance_ratio_clip'],
            clip_vf=params['clip_vf'],

            store=store,
            advanced_logging=advanced_logging,
            log_interval=log_interval,
            save_interval=params['save_interval'],

            seed=seed,
            cpu=not use_gpu,
            dtype=dtype
        )

        return p
