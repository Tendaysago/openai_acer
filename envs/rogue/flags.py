
from baselines.acer.flags import AcerFlags


class RogueFlags(AcerFlags):

    CFG_sections = {'RNG', 'Training', 'Log', 'Rogue'}

    def __init__(self):
        super().__init__()

        # Maximum number of steps in an episode
        self.max_episode_len = 500
        # Number of latest episodes to use for stats
        self.episodes_for_evaluation = 200

        # State generator name
        self.state_generator = "CroppedView_2L_17x17_StateGenerator"
        # Reward generator name
        self.reward_generator = "StairsOnly_RewardGenerator"

        # Wheter to enable monsters
        self.use_monsters = False
        # Whether to send the refresh command after each action
        self.refresh_after_commands = False
