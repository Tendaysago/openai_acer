#!/usr/bin/env python3
from baselines import logger
from baselines.acer.acer_simple import learn
from baselines.acer import models
from baselines.common.cmd_util import make_atari_env, atari_arg_parser
from baselines.acer.flags import AcerFlags


def main():
    parser = atari_arg_parser()
    parser.add_argument('--flags', '-f', help="flags cfg file", default=None)
    args = parser.parse_args()

    flags = AcerFlags.from_cfg(args.flags) if args.flags else AcerFlags()
    logger.configure(flags.log_dir)

    env = make_atari_env(args.env, num_env=flags.num_env, seed=flags.seed)

    policy_fn = models.get(args.policy)
    learn(policy_fn, env, flags)

    env.close()

if __name__ == '__main__':
    main()
