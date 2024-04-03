import torch as T

from Config import Config
from Environment import Environment
from ReplayMemory import ReplayMemory
from NNetworks.TestDQN import TestDQN

from copy import deepcopy


class Agent:
    def __init__(
        self, agent_config: Config, training_config: Config, environment: Environment
    ) -> None:
        self.agent_config = agent_config
        self.training_config = training_config

        self.state_space = self.agent_config.n_rays + 1
        self.n_actions = (
            self.agent_config.n_accelerations * self.agent_config.n_wheel_angles
        )

        self.Q_online = TestDQN(self.state_space, self.n_actions)
        self.Q_target = deepcopy(self.Q_online)

        self.memory = ReplayMemory(self.agent_config.memory_capacity)
        self.environment = environment

    def network_to_index(self, network_index):
        acc = network_index // self.agent_config.n_accelerations
        angle = network_index % self.agent_config.n_wheel_angles
        return acc, angle

    def index_to_action(self, grid_index): ...

    def choose_action(self, q_value): ...

    def learn_batch(self): ...

    def train(self): ...
