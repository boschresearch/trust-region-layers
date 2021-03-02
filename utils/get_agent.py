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

import git
import os
from cox.store import schema_from_dict

from trust_region_projections.algorithms.pg.pg import PolicyGradient
from trust_region_projections.utils.custom_store import CustomStore


def setup_general_agent(params: dict, save_git: bool = True):
    """
    General agent setup for logging results
    Args:
        params: dict with parameters
        save_git: Save git hash to restore experiment setting
    Returns:
         cox store for logging
    """
    for k, v in zip(params.keys(), params.values()):
        assert v is not None, f"Value for {k} is None"

    # ensure when not using entropy constraint, the cov is not projected to -inf by accident
    if not params['entropy_schedule']:
        params['entropy_eq'] = False

    store = None

    if params['log_interval'] <= params['train_steps']:
        # Setup logging
        metadata_schema = schema_from_dict(params)
        base_directory = params['out_dir']
        exp_name = params.get('exp_name')

        store = CustomStore(storage_folder=base_directory, exp_id=exp_name, new=True)

        # Store the experiment path
        metadata_schema.update({'store_path': str})
        metadata_table = store.add_table('metadata', metadata_schema)
        metadata_table.update_row(params)
        metadata_table.update_row({
            'store_path': store.path,
        })

        if save_git:
            # the git commit for this experiment
            metadata_schema.update({'git_commit': str})
            repo = git.Repo(path=os.path.dirname(os.path.realpath(__file__)), search_parent_directories=True)
            metadata_table.update_row({'git_commit': repo.head.object.hexsha})

        metadata_table.flush_row()

        # use 0 for saving last model only,
        # use -1 for no saving at all
        if params['save_interval'] == 0:
            params['save_interval'] = params['train_steps']

    return store


def get_new_ppo_agent(params, save_git=True):
    """
    Setup PPO specific logging
    Args:
        params: dict with parameters
        save_git: Save git hash to restore experiment setting
    Returns:
         PPO agent instance
    """
    store = setup_general_agent(params, save_git)

    if store:
        # Table for checkpointing models and envs
        if params['save_interval'] > 0:

            checkpoints_dict = {
                'policy': store.PYTORCH_STATE,
                'env_runner': store.PICKLE,
                'optimizer': store.PYTORCH_STATE,
                'iteration': int
            }

            if not params['share_weights'] and params["vf_coeff"] == 0:
                checkpoints_dict.update({'vf_model': store.PYTORCH_STATE,
                                         'optimizer_vf': store.PYTORCH_STATE
                                         })

            store.add_table('checkpoints', checkpoints_dict)
    else:
        store = None

    return PolicyGradient.agent_from_params(params, store=store)
