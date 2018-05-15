
from baselines.acer.flags import AcerFlags


class RogueAcerFlags(AcerFlags):

    CFG_sections = {'RNG', 'Training', 'Log', 'Rogue'}

    def __init__(self):
        super().__init__()

        self.policy = "Towers_LSTM"

        # It's important that nstack = 1
        self.nstack = 1

        self.nsteps = 60

        # Maximum number of steps in an episode
        self.max_episode_len = 500
        # Number of latest episodes to use for stats
        self.episodes_for_evaluation = 200

        # State generator name
        self.state_generator = "FullMap_5L_StateGenerator"
        # Reward generator name
        self.reward_generator = "StairsOnly_RewardGenerator"
        # Actions available
        self.actions = 'hjkl>'

        # Wheter to enable monsters
        self.use_monsters = False
        # Whether to send the refresh command after each action
        self.refresh_after_commands = False
