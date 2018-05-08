
from roguelib_module.rewards import RewardGenerator
from . import register


class StairsOnly_RewardGenerator(RewardGenerator):

    def get_value(self, frame_history):
        old_info = frame_history[-2]
        new_info = frame_history[-1]
        if new_info.statusbar["dungeon_level"] > old_info.statusbar["dungeon_level"]:
            self.goal_achieved = True
            return 10
        return 0


register(StairsOnly_RewardGenerator)
