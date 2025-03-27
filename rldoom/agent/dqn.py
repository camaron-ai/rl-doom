from torchtyping import TensorType as TT
from torch import nn


class DQN(nn.Module):
    def forward(self, x: TT['N', 'C', "W", "H"]) -> TT["N", "A"]:
        raise NotImplementedError


class DuelQNet(DQN):
    def __init__(self, available_actions_count: int, n_channles: int):
        super(DuelQNet, self).__init__()

        backbone = [
            nn.Conv2d(n_channles, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Flatten()
        ]
        
        self.backbone = nn.Sequential(*backbone)
        self.backbone_dim = 64 * 7 * 7
        hiddem_dim = 512
        self.state_fc = nn.Sequential(
            nn.Linear(self.backbone_dim, hiddem_dim),
            nn.ReLU(),
            nn.Linear(hiddem_dim, 1))

        self.advantage_fc = nn.Sequential(
            nn.Linear(self.backbone_dim, hiddem_dim),
            nn.ReLU(),
            nn.Linear(hiddem_dim, available_actions_count)
        )

    def forward(self, x: TT['N', 'C', "W", "H"]) -> TT["N", "A"]:
        x = self.backbone(x)
        state_value = self.state_fc(x)
        advantage_values = self.advantage_fc(x)
        x = state_value + (
            advantage_values - advantage_values.mean(dim=1).reshape(-1, 1)
        )
        return x

