
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


class StairsOnly_NthLevel_RewardGenerator(RewardGenerator):
    """Generates a reward of 10 whenever a higher level is reached.
    If 'objective_level' is reached, declares the game won.
    """

    objective_level = 10

    def reset(self):
        super().reset()
        self.last_level = 1

    def get_value(self, frame_history):
        last_frame = frame_history[-1]
        if last_frame.statusbar["dungeon_level"] > self.last_level:
            self.last_level = last_frame.statusbar["dungeon_level"]
            if self.last_level >= self.objective_level:
                self.goal_achieved = True
            return 10
        return 0


register(StairsOnly_NthLevel_RewardGenerator)
