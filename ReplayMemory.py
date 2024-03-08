from collections import deque, namedtuple
import random


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

test = Transition("state0", "action0", "next_state0", "reward0")


class ReplayMemory(object):
    def __init__(self, capacity) -> None:
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
