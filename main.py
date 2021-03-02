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

import argparse
import json
import logging
from glob import glob
from multiprocessing import JoinableQueue, Process

import os

from trust_region_projections.algorithms.pg.pg import PolicyGradient
from trust_region_projections.utils.custom_store import CustomStore
from utils.get_agent import get_new_ppo_agent


def multithreaded_run(agent_configs: str, agent_generator: callable, num_threads: int = 10):
    q = JoinableQueue()

    def run_single_config(queue):
        while True:
            conf_path = queue.get()
            params = json.load(open(conf_path))
            try:
                agent = agent_generator(params)
                agent.learn()
                agent.store.close()
            except Exception as e:
                logging.error("ERROR", e)
                raise e
            queue.task_done()

    for i in range(num_threads):
        worker = Process(target=run_single_config, args=(q,))
        worker.daemon = True
        worker.start()

    for fname in glob(os.path.join(agent_configs, "*.json")):
        q.put(fname)

    q.join()


def single_run(agent_config: str, agent_generator: callable):
    params = json.load(open(agent_config))

    # generate name
    params.update({
        "exp_name": f"{params['proj_type']}-"
                    f"{params['game']}-"
                    f"{params['policy_type']}-"
                    f"{'CONTEXT-' if params['contextual_std'] else ''}"
                    f"m{params['mean_bound']}-"
                    f"c{params['cov_bound']}-"
                    f"{'e' + str(params['target_entropy']) + '-' if params['entropy_schedule'] else ''}"
                    f"{'_' + str(params['entropy_schedule']) + '-' if params['entropy_schedule'] else ''}"
                    f"{'first' + str(params['entropy_first']) + '-' if params['entropy_schedule'] else ''}"
                    f"{'eq' + str(params['entropy_eq']) + '-' if params['entropy_schedule'] else ''}"
                    f"{'temp' + str(params['temperature']) + '-' if params['entropy_schedule'] else ''}"
                    f"lr{params['lr']}-"
                    f"lr_vf{params['lr_vf']}-"
                    f"{'lr_reg' + str(params['lr_reg']) + '-' if params['do_regression'] else ''}"
                    f"{'schedule' + str(params['lr_schedule']) + '-' if params['lr_schedule'] else ''}"
                    f"cov{params['init_std']}-"
                    f"min_std{params['minimal_std']}-"
                    f"{'delta' + str(params['trust_region_coeff']) + '-' if params['trust_region_coeff'] else ''}"
                    f"{'clip' + str(params['importance_ratio_clip']) + '-' if params['importance_ratio_clip'] else ''}"
                    f"{'max_ent' + str(params['max_entropy_coeff']) + '-' if params['max_entropy_coeff'] else ''}"
                    f"obs{params['norm_observations']}-"
                    f"{str(params['exp_name']) + '-' if params['exp_name'] else ''}"
                    f"steps{params['train_steps']}-"
                    f"epochs{params['epochs']}-"
                    f"seed{params['seed']}"
    })

    agent = agent_generator(params)
    agent.learn()
    agent.store.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run one or multiple runs for testing or plots.')
    parser.add_argument('path', type=str, help='Path to base config or root of experiment to load.')
    # parser.add_argument('--algorithm', type=str, default="pg", help='Specify which algorithm to use.')
    parser.add_argument('--load-exp-name', type=str, default=None, help='Load model from specified location.')
    parser.add_argument('--train-steps', type=int, default=None, help='New total training steps.', )
    parser.add_argument('--test', action='store_true', help='Only test loaded model.', )
    parser.add_argument('--num-threads', type=int, default=10,
                        help='Number of threads for running multiple experiments.', )
    args = parser.parse_args()

    path = args.path

    if args.load_exp_name:
        store = CustomStore(storage_folder=path, exp_id=args.load_exp_name, new=not args.test)
        agent, agent_params = PolicyGradient.agent_from_data(store, args.train_steps)
        if args.test:
            while True:
                _, eval_dict = agent.evaluate_policy(0, render=True, deterministic=True)
                print(eval_dict)
        else:
            agent.learn()
        agent.store.close()

    if not os.path.isfile(path):
        multithreaded_run(path, get_new_ppo_agent, num_threads=args.num_threads)
    else:
        single_run(path, get_new_ppo_agent)
