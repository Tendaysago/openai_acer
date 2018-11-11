import gym
import warnings

from baselines.common.cmd_util import arg_parser
from rogueinabox_lib.baseagent import RecordingWrapper
from rogueinabox_lib.randomagent import RandomAgent
from rogueinabox_lib.options import AgentOptions
from rogueinabox_lib.logger import Log

from envs.rogue import RogueAcerFlags, RogueEnv
from video_acer import ACER_Agent


class NstepsWrapper(RecordingWrapper):

    class RunIsOver(Exception):
        # raised when all n_steps are tested
        pass

    def __init__(self, wrappedAgent, eps_per_nsteps=1000, n_steps_list=[60]):
        self.stats = type('', (), {})()

        self.stats.eps_per_nsteps = eps_per_nsteps
        self.stats.n_steps_list = n_steps_list
        self.stats.current_nsteps = n_steps_list

        super().__init__(wrappedAgent, record_dir='video', reset_key='rR')

    def _new_episode(self):

        if self.episode_index != 0 and (self.episode_index % self.stats.eps_per_nsteps) == 0:
            stats = self.wrapped.rb.evaluator.statistics()
            self.logger.log([Log('stats', 'stats - nsteps=%s\n%s' % (self.stats.current_nsteps[0], str(stats)))])
            self.stats.current_nsteps.pop(0)
            if len(self.stats.current_nsteps) == 0:
                raise self.RunIsOver()
            self.wrapped.rb.evaluator.reset()
            self.wrapped.rb.evaluator.on_run_begin()
        super()._new_episode()

    def act(self):
        res = super().act()
        if (self.step_count % self.stats.current_nsteps[0]) == 0:
            self.reset_lstm_state()
        return res

    def reset_lstm_state(self):
        # set wrapped agent's lstm state to its initial value
        self.wrapped.policy_state = self.wrapped.policy.initial_state

    def record_screen(self):
        # disable frame recording
        pass    


if __name__ == '__main__':
    parser = arg_parser()
    parser.add_argument('cfg', help="cfg flags file")
    parser.add_argument('--checkpoint_path', '-c', help="checkpoint file to load")
    parser.add_argument('--eps_per_nsteps', '-e', type=int, help="episodes to play per nsteps value")
    parser.add_argument('--n_steps', '-ns', nargs='+', type=int, help="n_steps values to test")
    args = parser.parse_args()

    default_nsteps = [4, 8, 16, 32, 60]

    if args.n_steps is None:
        warnings.warn("n_steps not provided, using default value")
        args.n_steps = default_nsteps

    print("n_steps values:", args.n_steps)

    flags = RogueAcerFlags.from_cfg(args.cfg)

    try:
        RogueEnv.register(flags)
    except gym.error.Error:
        # an error is raised if Rogue was already registered
        pass

    flags.episodes_for_evaluation = args.eps_per_nsteps
    agent_opts = AgentOptions(gui=False)

    if args.checkpoint_path is None:
        raise ValueError("no checkpoint path provided")
    agent = ACER_Agent(agent_opts, flags=flags,
                       checkpoint_path=args.checkpoint_path)

    agent = NstepsWrapper(agent, eps_per_nsteps=args.eps_per_nsteps, n_steps_list=args.n_steps)
    try:
        agent.run()
    except NstepsWrapper.RunIsOver:
        # the agent played all the episodes for all n_steps values
        pass

