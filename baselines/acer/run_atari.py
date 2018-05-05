#!/usr/bin/env python3
from baselines import logger
from baselines.acer.acer_simple import learn
from baselines.acer import models
from baselines.common import tf_decay
from baselines.common.cmd_util import make_atari_env, atari_arg_parser


def main():
    parser = atari_arg_parser()
    parser.add_argument('--policy', help='Policy architecture', choices=models.registered_list(), default='CNN')
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=tf_decay.types(), default='constant')
    parser.add_argument('--logdir', help ='Directory for logging')
    args = parser.parse_args()
    logger.configure(args.logdir)

    env = make_atari_env(args.env, num_env=16, seed=args.seed)

    policy_fn = models.get(args.policy)
    learn(policy_fn, env, args.seed, total_timesteps=int(args.num_timesteps * 1.1), lrschedule=args.lrschedule)

    env.close()

if __name__ == '__main__':
    main()
