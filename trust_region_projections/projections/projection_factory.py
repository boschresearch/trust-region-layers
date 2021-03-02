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

from trust_region_projections.projections.base_projection_layer import BaseProjectionLayer
from trust_region_projections.projections.frob_projection_layer import FrobeniusProjectionLayer
from trust_region_projections.projections.kl_projection_layer import KLProjectionLayer
from trust_region_projections.projections.papi_projection import PAPIProjection
from trust_region_projections.projections.w2_projection_layer import WassersteinProjectionLayer


def get_projection_layer(proj_type: str = "", **kwargs) -> BaseProjectionLayer:
    """
    Factory to generate the projection layers for all projections.
    Args:
        proj_type: One of None/' ', 'ppo', 'papi', 'w2', 'w2_non_com', 'frob', 'kl', or 'entropy'
        **kwargs: arguments for projection layer

    Returns:

    """
    if not proj_type or proj_type.isspace() or proj_type.lower() in ["ppo", "sac", "td3", "mpo", "entropy"]:
        return BaseProjectionLayer(proj_type, **kwargs)

    elif proj_type.lower() == "w2":
        return WassersteinProjectionLayer(proj_type, **kwargs)

    elif proj_type.lower() == "frob":
        return FrobeniusProjectionLayer(proj_type, **kwargs)

    elif proj_type.lower() == "kl":
        return KLProjectionLayer(proj_type, **kwargs)

    elif proj_type.lower() == "papi":
        # papi has a different approach compared to our projections.
        # It has to be applied after the training with PPO.
        return PAPIProjection(proj_type, **kwargs)

    else:
        raise ValueError(
            f"Invalid projection type {proj_type}."
            f" Choose one of None/' ', 'ppo', 'papi', 'w2', 'w2_non_com', 'frob', 'kl', or 'entropy'.")
