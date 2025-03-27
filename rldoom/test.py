from .agent import Agent
import itertools
import torch
from vizdoom.vizdoom import DoomGame
import vizdoom as vzd
import time


class VisualizeAgent():
    def __init__(
        self,
        game: DoomGame,
        episodes_to_watch: int,
        frame_repeat: int,
        ):
        self.game = game
        self.episodes_to_watch = episodes_to_watch
        self.frame_repeat = frame_repeat

         # compute the number of possible actions
        total_buttons = game.get_available_buttons_size()
        self.ohe_actions = [list(a) for a in itertools.product([0, 1], repeat=total_buttons)]

    def run(self, agent: Agent):
        self.game.set_window_visible(True)
        self.game.set_mode(vzd.Mode.ASYNC_PLAYER)
        self.game.init()
        
        for _ in range(self.episodes_to_watch):
            self.game.new_episode()
            while not self.game.is_episode_finished():
                state = self.game.get_state()
                state_repr = agent.process_state(state)[torch.newaxis]
                best_action_index = agent.get_next_action(state_repr, 0.)
                action = self.ohe_actions[best_action_index]
                # Instead of make_action(a, frame_repeat) in order to make the animation smooth
                self.game.set_action(action)
                for _ in range(self.frame_repeat):
                    self.game.advance_action()

            # Sleep between episodes
            time.sleep(1.0)
            score = self.game.get_total_reward()
            print("Total score: ", score)
        
        self.game.close()