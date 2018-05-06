#!/usr/bin/env python3
import gym
from baselines import logger
from baselines.acer.acer_simple import learn
from baselines.acer import models
from baselines.common import set_global_seeds, tf_decay
from baselines.common.cmd_util import arg_parser

from envs import RogueEnv, RogueSubprocVecEnv


def main():
    parser = arg_parser()
    parser.add_argument('--num-timesteps', '-n', type=int, default=int(1e8))
    parser.add_argument('--policy', help='Policy architecture',
                        choices=models.registered_list(), default='CNN_LSTM')
    parser.add_argument('--lrschedule', help='Learning rate schedule',
                        choices=tf_decay.types(), default='constant')
    parser.add_argument('--logdir', help ='Directory for logging')
    args = parser.parse_args()
    flags = type('', (), {'max_episode_len': 500})()
    RogueEnv.register(flags)
    logger.configure(args.logdir)

    env = make_rogue_env(num_env=16)

    set_global_seeds(0)
    policy_fn = models.get(args.policy)
    learn(policy_fn, env, seed=0, nsteps=60, nstack=1, total_timesteps=int(args.num_timesteps * 1.1),
          lrschedule=args.lrschedule, save_dir='save', save_interval=100)

    env.close()


def make_rogue_env(num_env, start_index=0):
    def make_env(rank):
        def _thunk():
            env = gym.make('Rogue-v1')
            return env
        return _thunk
    return RogueSubprocVecEnv([make_env(i + start_index) for i in range(num_env)])


if __name__ == '__main__':
    main()
