
"""
Runs one instance of the environment and optimizes using the Soft Actor
Critic algorithm. Can use a GPU for the agent (applies to both sample and
train). No parallelism employed, everything happens in one python process; can
be easier to debug.

Requires OpenAI gym (and maybe mujoco).  If not installed, move on to next
example.

"""

import os
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.envs.gym import make as gym_make
from rlpyt.algos.qpg.ddpg import DDPG
from rlpyt.algos.qpg.preqn import PreQN
from rlpyt.agents.qpg.ddpg_agent import DdpgAgent
from rlpyt.agents.qpg.preqn_agent import PreqnAgent
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.logging.context import logger_context
from rlpyt.models.mlp import Sine, Linear
from rlpyt.utils.seed import set_seed
import torch


def build_and_train(env_id="HalfCheetah-v3", log_dir='results', alg_name='ddpg', run_ID=0, cuda_idx=None, seed=42, q_hidden_sizes=[64,64], q_nonlinearity='relu', batch_size=32, q_target=None, log_freq=1e3):
    set_seed(seed)
    sampler = SerialSampler(
        EnvCls=gym_make,
        env_kwargs=dict(id=env_id),
        eval_env_kwargs=dict(id=env_id),
        batch_T=1,  # One time-step per sampler iteration.
        batch_B=1,  # One environment (i.e. sampler Batch dimension).
        max_decorrelation_steps=0,
        eval_n_envs=10,
        eval_max_steps=int(51e3),
        eval_max_trajectories=50,
    )
    if q_nonlinearity == 'relu': q_nonlin = torch.nn.ReLU
    if q_nonlinearity == 'sine': q_nonlin = Sine
    if q_nonlinearity == 'linear': q_nonlin = Linear
    if alg_name.lower() == 'ddpg':
        if q_target is None: q_target = True
        algo = DDPG(batch_size=batch_size,
                    target=q_target, 
                    min_steps_learn=log_freq)
        agent = DdpgAgent(q_hidden_sizes=q_hidden_sizes, 
                           q_nonlinearity=q_nonlin)
    elif alg_name.lower() == 'preqn':
        if q_target is None: q_target = False
        algo = PreQN(batch_size=batch_size,
                    target=q_target, 
                    min_steps_learn=log_freq)
        agent = PreqnAgent(q_hidden_sizes=q_hidden_sizes, 
                           q_nonlinearity=q_nonlin)
    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        seed=seed,
        n_steps=1e6,
        log_interval_steps=log_freq, #1e4,
        affinity=dict(cuda_idx=cuda_idx),
    )
    config = dict(env_id=env_id)
    log_dir = os.path.join(log_dir, env_id)
    log_dir = os.path.join(log_dir, alg_name.lower())
    log_dir += '-' + q_nonlinearity
    log_dir += '-hs' + str(q_hidden_sizes)
    log_dir += '-qt' + str(q_target)
    log_dir += '-bs' + str(batch_size)
    name = '' #env_id
    with logger_context(log_dir, run_ID, name, config, override_prefix=True, use_summary_writer=True):
        runner.train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #parser.add_argument('--env_id', help='environment ID', default='HalfCheetah-v3')
    parser.add_argument('--env_id', help='environment ID', default='Swimmer-v2')
    parser.add_argument('--alg_name', help='algorithm to run', default='ddpg')
    parser.add_argument('--log_dir', help='log directory', default='/mnt/slow_ssd/erobb/rlclass')
    parser.add_argument('--run_ID', help='run ID (logging)', type=int, default=0)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=None)
    parser.add_argument('--qhsize', type=int, nargs='+', default=[64,64])
    parser.add_argument('--qnonlin', type=str, default='relu')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--qtarget', type=bool, default=None)
    args = parser.parse_args()
    build_and_train(
        env_id=args.env_id,
        log_dir=args.log_dir,
        alg_name=args.alg_name,
        run_ID=args.run_ID,
        cuda_idx=args.cuda_idx,
        q_hidden_sizes=args.qhsize,
        q_nonlinearity=args.qnonlin,
        batch_size=args.batch_size,
        q_target=args.qtarget,
    )
