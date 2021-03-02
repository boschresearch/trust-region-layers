import cpp_projection
import numpy as np
import torch as ch
from typing import Any, Tuple

from trust_region_projections.models.policy.abstract_gaussian_policy import AbstractGaussianPolicy
from trust_region_projections.projections.base_projection_layer import BaseProjectionLayer, mean_projection
from trust_region_projections.utils.projection_utils import gaussian_kl
from trust_region_projections.utils.torch_utils import get_numpy


class KLProjectionLayer(BaseProjectionLayer):

    def _trust_region_projection(self, policy: AbstractGaussianPolicy, p: Tuple[ch.Tensor, ch.Tensor],
                                 q: Tuple[ch.Tensor, ch.Tensor], eps: ch.Tensor, eps_cov: ch.Tensor, **kwargs):
        """
        Runs KL projection layer and constructs cholesky of covariance
        Args:
            policy: policy instance
            p: current distribution
            q: old distribution
            eps: (modified) kl bound/ kl bound for mean part
            eps_cov: (modified) kl bound for cov part
            **kwargs:

        Returns:
            projected mean, projected cov cholesky
        """
        mean, std = p
        old_mean, old_std = q

        if not policy.contextual_std:
            # only project first one to reduce number of numerical optimizations
            std = std[:1]
            old_std = old_std[:1]

        ################################################################################################################
        # project mean with closed form
        mean_part, _ = gaussian_kl(policy, p, q)
        proj_mean = mean_projection(mean, old_mean, mean_part, eps)

        cov = policy.covariance(std)
        old_cov = policy.covariance(old_std)

        if policy.is_diag:
            proj_cov = KLProjectionGradFunctionDiagCovOnly.apply(cov.diagonal(dim1=-2, dim2=-1),
                                                                 old_cov.diagonal(dim1=-2, dim2=-1),
                                                                 eps_cov)
            proj_std = proj_cov.sqrt().diag_embed()
        else:
            raise NotImplementedError("The KL projection currently does not support full covariance matrices.")

        if not policy.contextual_std:
            # scale first std back to batchsize
            proj_std = proj_std.expand(mean.shape[0], -1, -1)

        return proj_mean, proj_std


class KLProjectionGradFunctionDiagCovOnly(ch.autograd.Function):
    projection_op = None

    @staticmethod
    def get_projection_op(batch_shape, dim, max_eval=100):
        if not KLProjectionGradFunctionDiagCovOnly.projection_op:
            KLProjectionGradFunctionDiagCovOnly.projection_op = \
                cpp_projection.BatchedDiagCovOnlyProjection(batch_shape, dim, max_eval=max_eval)
        return KLProjectionGradFunctionDiagCovOnly.projection_op

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        std, old_std, eps_cov = args

        batch_shape = std.shape[0]
        dim = std.shape[-1]

        cov_np = get_numpy(std)
        old_std = get_numpy(old_std)
        eps = get_numpy(eps_cov) * np.ones(batch_shape)

        # p_op = cpp_projection.BatchedDiagCovOnlyProjection(batch_shape, dim)
        # ctx.proj = projection_op

        p_op = KLProjectionGradFunctionDiagCovOnly.get_projection_op(batch_shape, dim)
        ctx.proj = p_op

        proj_std = p_op.forward(eps, old_std, cov_np)

        return std.new(proj_std)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        projection_op = ctx.proj
        d_std, = grad_outputs

        d_std_np = get_numpy(d_std)
        d_std_np = np.atleast_2d(d_std_np)
        df_stds = projection_op.backward(d_std_np)
        df_stds = np.atleast_2d(df_stds)

        return d_std.new(df_stds), None, None
