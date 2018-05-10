#!/usr/bin/env python3
import gym
from baselines import logger
from baselines.acer.acer_simple import learn
from baselines.acer import models
from baselines.common import set_global_seeds, tf_decay
from baselines.common.cmd_util import arg_parser

from envs import RogueEnv, RogueSubprocVecEnv, RogueAcerFlags


def main():
    parser = arg_parser()
    parser.add_argument('--flags', '-f', help="flags cfg file", default=None)
    args = parser.parse_args()

    flags = RogueAcerFlags.from_cfg(args.flags) if args.flags else RogueAcerFlags()
    RogueEnv.register(flags)
    logger.configure(flags.log_dir)

    env = make_rogue_env(num_env=flags.num_env)

    set_global_seeds(flags.seed)
    policy_fn = models.get(flags.policy)
    learn(policy_fn, env, flags)

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
