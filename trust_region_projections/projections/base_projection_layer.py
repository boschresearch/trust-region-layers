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

import copy
import math
import torch as ch
from typing import Tuple, Union

from trust_region_projections.models.policy.abstract_gaussian_policy import AbstractGaussianPolicy
from trust_region_projections.utils.network_utils import get_optimizer
from trust_region_projections.utils.projection_utils import gaussian_kl, get_entropy_schedule
from trust_region_projections.utils.torch_utils import generate_minibatches, select_batch, tensorize


def entropy_inequality_projection(policy: AbstractGaussianPolicy, p: Tuple[ch.Tensor, ch.Tensor],
                                  beta: Union[float, ch.Tensor]):
    """
    Projects std to satisfy an entropy INEQUALITY constraint.
    Args:
        policy: policy instance
        p: current distribution
        beta: target entropy for EACH std or general bound for all stds

    Returns:
        projected std that satisfies the entropy bound
    """
    mean, std = p
    k = std.shape[-1]
    batch_shape = std.shape[:-2]

    ent = policy.entropy(p)
    mask = ent < beta

    # if nothing has to be projected skip computation
    if (~mask).all():
        return p

    alpha = ch.ones(batch_shape, dtype=std.dtype, device=std.device)
    alpha[mask] = ch.exp((beta[mask] - ent[mask]) / k)

    proj_std = ch.einsum('ijk,i->ijk', std, alpha)
    return mean, ch.where(mask[..., None, None], proj_std, std)


def entropy_equality_projection(policy: AbstractGaussianPolicy, p: Tuple[ch.Tensor, ch.Tensor],
                                beta: Union[float, ch.Tensor]):
    """
    Projects std to satisfy an entropy EQUALITY constraint.
    Args:
        policy: policy instance
        p: current distribution
        beta: target entropy for EACH std or general bound for all stds

    Returns:
        projected std that satisfies the entropy bound
    """
    mean, std = p
    k = std.shape[-1]

    ent = policy.entropy(p)
    alpha = ch.exp((beta - ent) / k)
    proj_std = ch.einsum('ijk,i->ijk', std, alpha)
    return mean, proj_std


def mean_projection(mean: ch.Tensor, old_mean: ch.Tensor, maha: ch.Tensor, eps: ch.Tensor):
    """
    Projects the mean based on the Mahalanobis objective and trust region.
    Args:
        mean: current mean vectors
        old_mean: old mean vectors
        maha: Mahalanobis distance between the two mean vectors
        eps: trust region bound

    Returns:
        projected mean that satisfies the trust region
    """
    batch_shape = mean.shape[:-1]
    mask = maha > eps

    ################################################################################################################
    # mean projection maha

    # if nothing has to be projected skip computation
    if mask.any():
        omega = ch.ones(batch_shape, dtype=mean.dtype, device=mean.device)
        omega[mask] = ch.sqrt(maha[mask] / eps) - 1.
        omega = ch.max(-omega, omega)[..., None]

        m = (mean + omega * old_mean) / (1 + omega + 1e-16)
        proj_mean = ch.where(mask[..., None], m, mean)
    else:
        proj_mean = mean

    return proj_mean


class BaseProjectionLayer(object):

    def __init__(self,
                 proj_type: str = "",
                 mean_bound: float = 0.03,
                 cov_bound: float = 1e-3,
                 trust_region_coeff: float = 0.0,
                 scale_prec: bool = True,

                 entropy_schedule: Union[None, str] = None,
                 action_dim: Union[None, int] = None,
                 total_train_steps: Union[None, int] = None,
                 target_entropy: float = 0.0,
                 temperature: float = 0.5,
                 entropy_eq: bool = False,
                 entropy_first: bool = False,

                 do_regression: bool = False,
                 regression_iters: int = 1000,
                 regression_lr: int = 3e-4,
                 optimizer_type_reg: str = "adam",

                 cpu: bool = True,
                 dtype: ch.dtype = ch.float32,
                 ):

        """
        Base projection layer, which can be used to compute metrics for non-projection approaches.
        Args:
           proj_type: Which type of projection to use. None specifies no projection and uses the TRPO objective.
           mean_bound: projection bound for the step size w.r.t. mean
           cov_bound: projection bound for the step size w.r.t. covariance matrix
           trust_region_coeff: Coefficient for projection regularization loss term.
           scale_prec: If true used mahalanobis distance for projections instead of euclidean with Sigma_old^-1.
           entropy_schedule: Schedule type for entropy projection, one of 'linear', 'exp', None.
           action_dim: number of action dimensions to scale exp decay correctly.
           total_train_steps: total number of training steps to compute appropriate decay over time.
           target_entropy: projection bound for the entropy of the covariance matrix
           temperature: temperature decay for exponential entropy bound
           entropy_eq: Use entropy equality constraints.
           entropy_first: Project entropy before trust region.
           do_regression: Conduct additional regression steps after the the policy steps to match projection and policy.
           regression_iters: Number of regression steps.
           regression_lr: Regression learning rate.
           optimizer_type_reg: Optimizer for regression.
           cpu: Compute on CPU only.
           dtype: Data type to use, either of float32 or float64. The later might be necessary for higher
                   dimensions in order to learn the full covariance.
        """

        # projection and bounds
        self.proj_type = proj_type
        self.mean_bound = tensorize(mean_bound, cpu=cpu, dtype=dtype)
        self.cov_bound = tensorize(cov_bound, cpu=cpu, dtype=dtype)
        self.trust_region_coeff = trust_region_coeff
        self.scale_prec = scale_prec

        # projection utils
        assert (action_dim and total_train_steps) if entropy_schedule else True
        self.entropy_proj = entropy_equality_projection if entropy_eq else entropy_inequality_projection
        self.entropy_schedule = get_entropy_schedule(entropy_schedule, total_train_steps, dim=action_dim)
        self.target_entropy = tensorize(target_entropy, cpu=cpu, dtype=dtype)
        self.entropy_first = entropy_first
        self.entropy_eq = entropy_eq
        self.temperature = temperature
        self._initial_entropy = None

        # regression
        self.do_regression = do_regression
        self.regression_iters = regression_iters
        self.lr_reg = regression_lr
        self.optimizer_type_reg = optimizer_type_reg

    def __call__(self, policy, p: Tuple[ch.Tensor, ch.Tensor], q, step, *args, **kwargs):
        # entropy_bound = self.policy.entropy(q) - self.target_entropy
        entropy_bound = self.entropy_schedule(self.initial_entropy, self.target_entropy, self.temperature,
                                              step) * p[0].new_ones(p[0].shape[0])
        return self._projection(policy, p, q, self.mean_bound, self.cov_bound, entropy_bound, **kwargs)

    def _trust_region_projection(self, policy: AbstractGaussianPolicy, p: Tuple[ch.Tensor, ch.Tensor],
                                 q: Tuple[ch.Tensor, ch.Tensor], eps: ch.Tensor, eps_cov: ch.Tensor, **kwargs):
        """
        Hook for implementing the specific trust region projection
        Args:
            policy: policy instance
            p: current distribution
            q: old distribution
            eps: mean trust region bound
            eps_cov: covariance trust region bound
            **kwargs:

        Returns:
            projected
        """
        return p

    # @final
    def _projection(self, policy: AbstractGaussianPolicy, p: Tuple[ch.Tensor, ch.Tensor],
                    q: Tuple[ch.Tensor, ch.Tensor], eps: ch.Tensor, eps_cov: ch.Tensor, beta: ch.Tensor, **kwargs):
        """
        Template method with hook _trust_region_projection() to encode specific functionality.
        (Optional) entropy projection is executed before or after as specified by entropy_first.
        Do not override this. For Python >= 3.8 you can use the @final decorator to enforce not overwriting.
        Args:
            policy: policy instance
            p: current distribution
            q: old distribution
            eps: mean trust region bound
            eps_cov: covariance trust region bound
            beta: entropy bound
            **kwargs:

        Returns:
            projected mean, projected std
        """

        ####################################################################################################################
        # entropy projection in the beginning
        if self.entropy_first:
            p = self.entropy_proj(policy, p, beta)

        ####################################################################################################################
        # trust region projection for mean and cov bounds
        proj_mean, proj_std = self._trust_region_projection(policy, p, q, eps, eps_cov, **kwargs)

        ####################################################################################################################
        # entropy projection in the end
        if self.entropy_first:
            return proj_mean, proj_std

        return self.entropy_proj(policy, (proj_mean, proj_std), beta)

    @property
    def initial_entropy(self):
        return self._initial_entropy

    @initial_entropy.setter
    def initial_entropy(self, entropy):
        if self.initial_entropy is None:
            self._initial_entropy = entropy

    def trust_region_value(self, policy, p, q):
        """
        Computes the KL divergence between two Gaussian distributions p and q.
        Args:
            policy: policy instance
            p: current distribution
            q: old distribution
        Returns:
            Mean and covariance part of the trust region metric.
        """
        return gaussian_kl(policy, p, q)

    def get_trust_region_loss(self, policy: AbstractGaussianPolicy, p: Tuple[ch.Tensor, ch.Tensor],
                              proj_p: Tuple[ch.Tensor, ch.Tensor]):
        """
        Compute the trust region loss to ensure policy output and projection stay close.
        Args:
            policy: policy instance
            proj_p: projected distribution
            p: predicted distribution from network output

        Returns:
            trust region loss
        """
        p_target = (proj_p[0].detach(), proj_p[1].detach())
        mean_diff, cov_diff = self.trust_region_value(policy, p, p_target)

        delta_loss = (mean_diff + cov_diff if policy.contextual_std else mean_diff).mean()

        return delta_loss * self.trust_region_coeff

    def compute_metrics(self, policy, p, q) -> dict:
        """
        Returns dict with constraint metrics.
        Args:
            policy: policy instance
            p: current distribution
            q: old distribution

        Returns:
            dict with constraint metrics
        """
        with ch.no_grad():
            entropy_old = policy.entropy(q)
            entropy = policy.entropy(p)
            mean_kl, cov_kl = gaussian_kl(policy, p, q)
            kl = mean_kl + cov_kl

            mean_diff, cov_diff = self.trust_region_value(policy, p, q)

            combined_constraint = mean_diff + cov_diff
            entropy_diff = entropy_old - entropy

        return {'kl': kl.detach().mean(),
                'constraint': combined_constraint.mean(),
                'mean_constraint': mean_diff.mean(),
                'cov_constraint': cov_diff.mean(),
                'entropy': entropy.mean(),
                'entropy_diff': entropy_diff.mean(),
                'kl_max': kl.max(),
                'constraint_max': combined_constraint.max(),
                'mean_constraint_max': mean_diff.max(),
                'cov_constraint_max': cov_diff.max(),
                'entropy_max': entropy.max(),
                'entropy_diff_max': entropy_diff.max()
                }

    def trust_region_regression(self, policy: AbstractGaussianPolicy, obs: ch.Tensor, q: Tuple[ch.Tensor, ch.Tensor],
                                n_minibatches: int, global_steps: int):
        """
        Take additional regression steps to match projection output and policy output.
        The policy parameters are updated in-place.
        Args:
            policy: policy instance
            obs: collected observations from trajectories
            q: old distributions
            n_minibatches: split the rollouts into n_minibatches.
            global_steps: current number of steps, required for projection
        Returns:
            dict with mean of regession loss
        """

        if not self.do_regression:
            return {}

        policy_unprojected = copy.deepcopy(policy)
        optim_reg = get_optimizer(self.optimizer_type_reg, policy_unprojected.parameters(), learning_rate=self.lr_reg)
        optim_reg.reset()

        reg_losses = obs.new_tensor(0.)

        # get current projected values --> targets for regression
        p_flat = policy(obs)
        p_target = self(policy, p_flat, q, global_steps)

        for _ in range(self.regression_iters):
            batch_indices = generate_minibatches(obs.shape[0], n_minibatches)

            # Minibatches SGD
            for indices in batch_indices:
                batch = select_batch(indices, obs, p_target[0], p_target[1])
                b_obs, b_target_mean, b_target_std = batch
                proj_p = (b_target_mean.detach(), b_target_std.detach())

                p = policy_unprojected(b_obs)

                # invert scaling with coeff here as we do not have to balance with other losses
                loss = self.get_trust_region_loss(policy, p, proj_p) / self.trust_region_coeff

                optim_reg.zero_grad()
                loss.backward()
                optim_reg.step()
                reg_losses += loss.detach()

        policy.load_state_dict(policy_unprojected.state_dict())

        if not policy.contextual_std:
            # set policy with projection value.
            # In non-contextual cases we have only one cov, so the projection is the same.
            policy.set_std(p_target[1][0])

        steps = self.regression_iters * (math.ceil(obs.shape[0] / n_minibatches))
        return {"regression_loss": (reg_losses / steps).detach()}
