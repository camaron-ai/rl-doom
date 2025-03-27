import random
from torchtyping import TensorType as TT
import torch
from .core import Transition, TransitionBatch, Memory
from typing import List


class PerMemory(Memory):
    def __init__(
            self,
            state_dim: List[int],
            capacity: int,
            alpha: float = 0.7,
            beta: float = 1.,
            ):
        self.capacity = capacity
        self.alpha = alpha 
        self.size = 0
        self.next_idx = 0
        self.max_priority = 1. 

        self.memory = {
            "states": torch.zeros(self.capacity, *state_dim, dtype=torch.float32),
            "actions": torch.zeros(self.capacity, 1, dtype=torch.long),
            "rewards": torch.zeros(self.capacity, 1, dtype=torch.float32),
            "next_states": torch.zeros(self.capacity, *state_dim, dtype=torch.float32),
            "has_next_state_mask": torch.zeros(self.capacity, dtype=torch.bool)
        }
        self.priority_sum = [0 for _ in range(2 * self.capacity)]
        self.priority_min = [float("inf") for _ in range(2 * self.capacity)]
        self.beta = beta


    def push(self, transition: Transition):
        idx = self.next_idx
        self.memory["states"][idx] = transition.state
        self.memory["actions"][idx] = transition.action
        self.memory["rewards"][idx] = transition.reward
        self.memory["next_states"][idx] = transition.next_state
        self.memory["has_next_state_mask"][idx] = not transition.done
        
        
        self.next_idx = (idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        priority_alpha = self.max_priority ** self.alpha
        self._set_priority(idx, priority_alpha)
    

    def _set_priority(self, idx: int, priority_alpha: float):
        i = idx + self.capacity
        self.priority_sum[i] = priority_alpha
        self.priority_min[i] = priority_alpha
        # while not root
        while i >= 2:
            # get to my parent
            i //= 2
            self.priority_sum[i] = self.priority_sum[2 * i] + self.priority_sum[2 * i + 1]
            self.priority_min[i] = min(self.priority_min[2 * i], self.priority_min[2 * i + 1])
    

    def find_prefix_sum(self, prefix_sum: float) -> None:
        i = 1
        while i < self.capacity:
            if self.priority_sum[2 * i] > prefix_sum:
                # go left
                i = 2 * i
            else:
                # go right
                prefix_sum -= self.priority_sum[2 * i]
                i = 2 * i + 1
        
        return i - self.capacity 
    
    def sum(self) -> float:
        return self.priority_sum[1]
 
    def min(self) -> float:
        return self.priority_min[1]
 
    def sample(self, batch_size: int) -> TransitionBatch:
        indices = torch.zeros(batch_size, dtype=torch.long)
        weights = torch.zeros(batch_size, 1, dtype=torch.float32)
        
        for i in range(batch_size):
            prefix_sum = random.random() * self.sum()
            indices[i] = self.find_prefix_sum(prefix_sum)

        # minimum prob in the entire memory
        min_prob = self.min() / self.sum() 

        # maximum weight 
        max_weight = (min_prob * self.size) ** (-self.beta)
        for i in range(batch_size):
            idx = indices[i]
            prob = self.priority_sum[idx + self.capacity] / self.sum()
            
            weight = (prob * self.size) ** (-self.beta)
            weights[i, 0] = weight / max_weight
        

        return TransitionBatch(
            states=self.memory["states"][indices],
            actions=self.memory["actions"][indices],
            rewards=self.memory["rewards"][indices],
            next_states=self.memory["next_states"][indices],
            has_next_state_mask=self.memory["has_next_state_mask"][indices],
            weights=weights,
            indices=indices,
        )
        
    def update(self, indices: TT['N'], priorities: TT['N']) -> None:
        for idx, prior in zip(indices, priorities):
            prior = prior.item()
            assert isinstance(prior, float)
            assert prior > 0.
            self.max_priority = max(self.max_priority, prior)
            priority_alpha = prior ** self.alpha
            self._set_priority(idx, priority_alpha)

    def __len__(self) -> int:
        return self.size


