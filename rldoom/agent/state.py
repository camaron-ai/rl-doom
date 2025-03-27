from typing import Protocol, Tuple
from vizdoom.vizdoom import GameState
import torch
from torchvision import transforms
import numpy as np
from torchtyping import TensorType as TT


class StateProcessor(Protocol):
    def __call__(self, state: GameState) -> torch.Tensor:
        raise NotImplementedError


class V1StateProcessor:
    def __init__(self, size: Tuple[int, int]):
        self.processor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(size),
            ]
        ) 
    
    def __call__(self, state: GameState) -> TT['C', "W", "H"]:
        state_repr = np.stack([state.screen_buffer, state.depth_buffer])
        # from channel, width, height to width, height, channel
        state_repr = state_repr.transpose((1, 2, 0))
        return self.processor(state_repr)