import torch.nn as nn
import torch.nn.functional as F


class TestDQN(nn.Module):
    def __init__(self, n_observations, n_actions) -> None:
        super(TestDQN, self).__init__()
        self.fc1 = nn.Linear(n_observations, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
