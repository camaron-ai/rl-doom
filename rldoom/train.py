from .agent import Agent
from .memory import Transition, PerMemory
import itertools
import torch
from tqdm.auto import tqdm
from vizdoom.vizdoom import DoomGame
import os
from typing import List, Dict
from collections import defaultdict
import numpy as np
import pprint
import shutil
from .util import seed_everything


class Trainer():
    def __init__(
        self,
        logdir: str,
        game: DoomGame,
        memory: PerMemory,
        num_epochs: int,
        steps_per_epoch: int,
        seed: int,
        frame_repeat: int = 12,
        log_every_k_epochs: int = 10,
        init_epsilon: float = 1.,
        epsilon_decay: float = 0.9996,  
        min_epsilon: float = 0.1,
        ):
        self.game = game
        self.memory = memory
        self.logdir = logdir

        self.seed = seed
        self.steps_per_epoch = steps_per_epoch
        self.num_epochs = num_epochs

        self.frame_repeat = frame_repeat
        self.pbar = tqdm(total=self.steps_per_epoch)


        # compute the number of possible actions
        total_buttons = game.get_available_buttons_size()
        self.ohe_actions = [list(a) for a in itertools.product([0, 1], repeat=total_buttons)]

        assert len(self.ohe_actions) == 2 ** total_buttons

        self.log_every_k_epochs = log_every_k_epochs

        # epsilon decay
        self.epsilon = init_epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.epoch = 0
        self.step = 0

        # scale for the magnitud, this could be improve.
        self.reward_scale = 1 / 100


    def train_epoch(self, agent: Agent):
        # start new episode
        self.game.new_episode()
        self.pbar.reset()
        scores: Dict[str, List[float]] = defaultdict(list)

        for _ in range(self.steps_per_epoch):
            # get current state 
            state = self.game.get_state()
            # process
            state_repr = agent.process_state(state)[torch.newaxis]
            # get the next action
            action = agent.get_next_action(state_repr, self.epsilon)

            # action to right format
            ohe_action = self.ohe_actions[action.item()]

            # make action and collect step
            reward = self.game.make_action(ohe_action, self.frame_repeat)
            done = self.game.is_episode_finished()

            # get the next state is not done
            next_state_repr = (
                agent.process_state(self.game.get_state())[torch.newaxis]
                if not done else
                torch.zeros_like(state_repr)
            )
            # scale rewards
            reward = torch.tensor([[reward * self.reward_scale]], dtype=torch.float32)        
            # create transition
            transition = Transition(state_repr, action, reward, next_state_repr, done)

            # add transition
            self.memory.push(transition)

            if self.step >= agent.batch_size:
                # update agent
                loss = agent.train(self.memory)
                # udpate epsilon
                self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
                # keep track of the score
                scores["loss"].append(loss)


            if done:
                # track the total reward of the episode
                total_reward = self.game.get_total_reward()
                scores["reward"].append(total_reward)
                # restart game
                self.game.new_episode()

            self.pbar.set_postfix(epsilon=self.epsilon)
            self.step += 1
            self.pbar.update()

        avg_scores = {
            name: np.mean(ss) for name, ss in scores.items() 
        }
        pprint.pprint(avg_scores) 

        # update the target net after each epoch
        agent.update_target_net()
        output_path = os.path.join(self.logdir, f"policy_E{self.epoch:05d}.ckpt")
        agent.save(output_path)


    def train(self, agent: Agent):


        self.game.init()
        # remove the previous checkpoint
        if os.path.exists(self.logdir):
            shutil.rmtree(self.logdir)
        os.makedirs(self.logdir, exist_ok=True)

        seed_everything(self.seed)
        for _ in range(self.num_epochs):
            self.train_epoch(agent)   
            self.epoch += 1
        self.game.close()