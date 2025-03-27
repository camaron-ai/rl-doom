from torchtyping import TensorType as TT
import torch
from torch import nn, optim
import numpy as np
from ..memory import PerMemory
from .dqn import DQN
from .state import StateProcessor


def update_target_from_policy(target: nn.Module, policy: nn.Module): 
    policy_state_dict = policy.state_dict()
    target.load_state_dict(policy_state_dict)


class Agent:
    def __init__(
        self,
        n_actions: int,
        state_processor: StateProcessor,
        policy_net: DQN,
        target_net: DQN,
        batch_size: int,
        discount_factor: float,
        lr: float,
        device: torch.device
                 ):
        self.state_processor = state_processor
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.lr = lr
        self.grad_clip = 100
        self.policy_net = policy_net.to(device)
        self.target_net = target_net.to(device)
        
        update_target_from_policy(self.target_net, self.policy_net)

        self.device = device
        self.optimizer = self.configure_optimizer()


    def configure_optimizer(self) -> optim.Optimizer:
        return optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)

    def process_state(self, state) -> TT['C', "W", 'H']:
        return self.state_processor(state)

    @torch.no_grad
    def get_next_action(
        self,
        state: torch.Tensor,
        epsilon: float):
        # choose a random action
        if np.random.random() < epsilon:
            random_action = np.random.choice(self.n_actions)
            return torch.tensor([[random_action]], dtype=torch.long)

        state = state.to(self.device)
        # choose policy best action
        return torch.argmax(self.policy_net(state), dim=1, keepdim=True).cpu()

 
    def train(self, memory: PerMemory) -> float:
        # sample from the memory
        batch = memory.sample(self.batch_size)
        # move batch to the current device
        batch = batch.to(self.device)

        # get q values for the current state
        state_qvalues: TT['N', 1] = self.policy_net(batch.states).gather(1, batch.actions)
        # dual dqn
        with torch.no_grad():
            future_rewards = torch.zeros_like(batch.rewards)
            valid_next_states = batch.next_states[batch.has_next_state_mask] 
            # choose next action with policy
            next_actions = torch.argmax(self.policy_net(valid_next_states), dim=1, keepdim=True)
            # take the q values of target net
            future_rewards[batch.has_next_state_mask] = self.target_net(valid_next_states).gather(1, next_actions)
            assert future_rewards.shape == batch.rewards.shape
            expected_reward = batch.rewards + self.discount_factor * future_rewards
        assert state_qvalues.shape == expected_reward.shape

        # compute time difference
        td: TT["N"] = expected_reward - state_qvalues 

        # the loss is weighted MSE 
        loss = (batch.weights * td.pow(2)).mean()
        # backprop
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), self.grad_clip)
        self.optimizer.step()

        # priorities based on time difference magnitude 
        # + 1e-5 to aviod priorities = 0
        priorities = td.abs().detach().cpu().squeeze() + 1e-5

        # update priorities for sampling
        memory.update(batch.indices.cpu(), priorities)
        return loss.item()

    def update_target_net(self):
        # full update
        update_target_from_policy(self.target_net, self.policy_net) 
    

    def save(self, output_path: str) -> None:
        policy_weights = self.policy_net.state_dict()
        torch.save(policy_weights, output_path)
    
