from typing import NamedTuple, Protocol
from torchtyping import TensorType as TT
import abc


class Transition(NamedTuple):
    state: TT[1, 'C', "W", "H"]
    action: TT[1, "A"]
    reward: TT[1, 1]
    next_state: TT[1, 'C', "W", "H"]
    done: bool


class TransitionBatch(NamedTuple):
    states: TT["N", "C", "W", "H"]
    actions: TT["N", "A"]
    rewards: TT["N", 1]
    next_states: TT["N", "C", "W", "H"]
    has_next_state_mask: TT["N"]
    indices: TT["N"]
    weights: TT['N'] 
   
    def to(self, device: str) -> "TransitionBatch":
        return TransitionBatch(*[t.to(device) for t in self])


class Memory(Protocol):
    @abc.abstractmethod
    def __len__(self):
        raise NotImplementedError
    
    @abc.abstractmethod
    def push(self, transition: Transition):
        raise NotImplementedError
    
    @abc.abstractmethod
    def sample(self, batch_size: int) -> TransitionBatch:
        raise NotImplementedError