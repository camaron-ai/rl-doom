import random
from collections import deque
import torch
from .core import Memory, Transition, TransitionBatch

 
class ReplayMemory(Memory):
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)
    

    def push(self, transition: Transition):
        self.memory.append(transition)
    

    def sample(self, batch_size: int) -> TransitionBatch:
        transitions = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)

        batch_states = torch.cat(states)
        batch_actions = torch.cat(actions)
        batch_rewards = torch.cat(rewards)


        has_next_state_mask = torch.tensor(dones, dtype=torch.bool)
        batch_next_states = torch.cat(next_states)

        return TransitionBatch(
            states=batch_states,
            actions=batch_actions,
            rewards=batch_rewards,
            has_next_state_mask=has_next_state_mask,
            next_states=batch_next_states,
            indices=None,
            weights=None
        )

    def __len__(self) -> int:
        return len(self.memory)

