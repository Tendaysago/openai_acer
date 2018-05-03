
import os
import pickle
import collections
import warnings
import numpy as np
import gym.spaces

from roguelib_module.rogueinabox import RogueBox
from roguelib_module.states import CroppedView_SingleLayer_17x17_StateGenerator
from roguelib_module.rewards import RewardGenerator


EPISODES_TO_KEEP = 5


class RogueEnv(gym.Env):

    metadata = {'render.modes': ['ansi']}

    reward_range = (-np.inf, np.inf)
    actions = RogueBox.get_actions()
    action_space = gym.spaces.Discrete(len(actions))
    observation_space = gym.spaces.Box(low=0, high=32, shape=(17, 17, 2), dtype=np.float)

    @classmethod
    def register(cls, FLAGS):
        gym.envs.register('Rogue-v1',
                          entry_point=RogueEnv, trials=200,
                          reward_threshold=None, local_only=True,
                          kwargs=None, nondeterministic=True,
                          tags=None, max_episode_steps=FLAGS.max_episode_len,
                          max_episode_seconds=None, timestep_limit=None)

    def __init__(self):
        self.rb = RogueBox(use_monsters=False,
                           max_step_count=500,
                           episodes_for_evaluation=200,
                           state_generator=CroppedView_2L_17x17_StateGenerator(),
                           reward_generator=StairsOnly_RewardGenerator(),
                           refresh_after_commands=False)
        self._saved_episodes = collections.deque()

    def render(self, mode='ansi'):
        if mode != 'ansi':
            raise NotImplementedError
        return '\n'.join(self.rb.screen) + '\n'

    def reset(self):
        self.rb.reset()
        state = np.expand_dims(self.rb.state, axis=0)
        return state

    def close(self):
        self.rb.stop()

    def step(self, action):
        """
        :param np.ndarray action:
            action to perform, represented as an integer in [0, len(actions)[

        :return:
            next_state, reward, done, info
        """
        command = self.actions[action]
        reward, new_state, won, lost = self.rb.send_command(command)
        return np.expand_dims(new_state, axis=0), reward, won or lost, self.rb.get_last_frame()

    def stats(self):
        return self.rb.evaluator.statistics()

    def save_state(self, path, id):
        self._save_episodes(path, id)

    def restore_state(self, path, id, warn=True):
        self._restore_episodes(path, id, warn=warn)

    def _episodes_path(self, checkpoint_dir, id):
        return os.path.join(checkpoint_dir, 'episodes', 'episodes-%s.pkl' % id)

    def _save_episodes(self, checkpoint_dir, id):
        os.makedirs(os.path.join(checkpoint_dir, 'episodes'), exist_ok=True)
        path = self._episodes_path(checkpoint_dir, id)
        with open(path, mode='wb') as pkfile:
            pickle.dump(self.rb.evaluator.episodes, pkfile)
        self._saved_episodes.append(path)
        if len(self._saved_episodes) > EPISODES_TO_KEEP:
            old_path = self._saved_episodes.popleft()
            try:
                os.unlink(old_path)
            except FileNotFoundError:
                warnings.warn('Attempting to delete unexisting episodes file %s: it was removed by an external program.'
                              % old_path, RuntimeWarning)

    def _restore_episodes(self, checkpoint_dir, id, warn=True):
        path = self._episodes_path(checkpoint_dir, id)
        try:
            with open(path, mode='rb') as pkfile:
                self.rb.evaluator.episodes = pickle.load(pkfile)
        except FileNotFoundError:
            if warn:
                warnings.warn('Episodes file %s not found: stats may be skewed.' % path, RuntimeWarning)


class CroppedView_2L_17x17_StateGenerator(CroppedView_SingleLayer_17x17_StateGenerator):

    def _set_shape(self, data_format):
        self._shape = (2, 17, 17) if data_format == "channels_first" else (17, 17, 2)

    def build_state(self, current_frame, frame_history):
        state = self.empty_state()

        self.set_channel_relative(self.player_position, state, 1, current_frame.get_list_of_positions_by_tile("%"), 4)  # stairs
        self.set_channel_relative(self.player_position, state, 0, current_frame.get_list_of_positions_by_tile("|"), 8)  # walls
        self.set_channel_relative(self.player_position, state, 0, current_frame.get_list_of_positions_by_tile("-"), 8)  # walls
        self.set_channel_relative(self.player_position, state, 0, current_frame.get_list_of_positions_by_tile("+"), 16)  # doors
        self.set_channel_relative(self.player_position, state, 0, current_frame.get_list_of_positions_by_tile("#"), 16)  # tunnel

        return state


class StairsOnly_RewardGenerator(RewardGenerator):

    def get_value(self, frame_history):
        old_info = frame_history[-2]
        new_info = frame_history[-1]
        if new_info.statusbar["dungeon_level"] > old_info.statusbar["dungeon_level"]:
            self.goal_achieved = True
            return 10
        return 0
