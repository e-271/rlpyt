
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


def build_and_train(env_id="HalfCheetah-v3", log_dir='results', alg_name='ddpg', run_ID=0, cuda_idx=None):
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
    if alg_name.lower() == 'ddpg':
        algo = DDPG()  # Run with defaults.
        agent = DdpgAgent()
    elif alg_name.lower() == 'preqn':
        algo = PreQN()  # Run with defaults.
        agent = PreqnAgent()
    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=1e6,
        log_interval_steps=1e4,
        affinity=dict(cuda_idx=cuda_idx),
    )
    config = dict(env_id=env_id)
    log_dir = os.path.join(log_dir, env_id)
    log_dir = os.path.join(log_dir, alg_name.lower())
    name = '' #env_id
    with logger_context(log_dir, run_ID, name, config, override_prefix=True, use_summary_writer=True):
        runner.train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env_id', help='environment ID', default='HalfCheetah-v3')
    parser.add_argument('--alg_name', help='algorithm to run', default='ddpg')
    parser.add_argument('--log_dir', help='log directory', default='/mnt/slow_ssd/erobb/rlclass')
    parser.add_argument('--run_ID', help='run ID (logging)', type=int, default=0)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=None)
    args = parser.parse_args()
    build_and_train(
        env_id=args.env_id,
        log_dir=args.log_dir,
        alg_name=args.alg_name,
        run_ID=args.run_ID,
        cuda_idx=args.cuda_idx,
    )
