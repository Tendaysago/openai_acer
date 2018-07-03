
from rogueinabox_lib.states import CroppedView_SingleLayer_17x17_StateGenerator
from . import register


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


register(CroppedView_2L_17x17_StateGenerator)
