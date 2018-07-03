
import gym
import tensorflow as tf

from baselines.acer import models
from baselines.common.cmd_util import arg_parser
from baselines.common import tf_util
from rogueinabox_lib.baseagent import BaseAgent, RecordingWrapper
from rogueinabox_lib.options import AgentOptions

from envs.rogue import RogueAcerFlags, RogueEnv


class ACER_Agent(BaseAgent):
    """
    UI ACER agent
    """

    def __init__(self, options, flags=RogueAcerFlags(), checkpoint_path=''):
        """
        :param AgentOptions options:
            rogueinabox AgentOptions
            N.B. roguebox_options are ignored, use flags
        :param RogueAcerFlags flags:
            flags to use
        :param str checkpoint_path:
            checkpoint path to be loaded
        """

        # disable gpu before creating any tensor
        tf_util.disable_gpu()

        self.sess = tf.Session()

        self.env = gym.make('Rogue-v1')

        PolicyModel = models.get(flags.policy)
        self.policy = PolicyModel(self.sess, self.env.observation_space, self.env.action_space,
                                  nenv=1, nsteps=1, nstack=1)  # type: models.Model

        if checkpoint_path:
            saver = tf.train.Saver()
            saver.restore(self.sess, checkpoint_path)
        else:
            self.sess.run(tf.global_variables_initializer())

        self.policy_state = self.policy.initial_state
        self.rogue_state = None

        super().__init__(options)

    def _create_rogue(self, configs):
        self.rogue_state = self.env.reset()
        return self.env.unwrapped.rb

    def game_over(self):
        self.rogue_state = self.env.reset()
        self.policy_state = self.policy.initial_state

    def act(self):
        [action], _mus, self.policy_state = self.policy.step([self.rogue_state], state=self.policy_state, mask=[False])
        self.rogue_state, _reward, done, _ = self.env.step(action)
        return done


def video(flags=RogueAcerFlags(), checkpoint_path=None, record_dir=None):
    try:
        RogueEnv.register(flags)
    except gym.error.Error:
        # an error is raised if Rogue was already registered
        pass

    options = AgentOptions(gui=True, gui_timer_ms=50, userinterface='curses')

    agent = ACER_Agent(options, flags=flags, checkpoint_path=checkpoint_path)

    if record_dir:
        agent = RecordingWrapper(agent, record_dir=record_dir)

    agent.run()


if __name__ == '__main__':
    parser = arg_parser()
    parser.add_argument('--flags', '-f', help="flags cfg file", default=None)
    parser.add_argument('--record_dir', '-r',
                        help="directory where to record frames on file (leave blank to avoid recording)",
                        default='')
    parser.add_argument('--checkpoint_path', '-c',
                        help="checkpoint file to load (without extension) N.B.: if you don't provide one an untrained model will be used")
    args = parser.parse_args()

    flags = RogueAcerFlags.from_cfg(args.flags) if args.flags else RogueAcerFlags()

    video(flags=flags, checkpoint_path=args.checkpoint_path, record_dir=args.record_dir)
