import random
import numpy as np

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.memory = []
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        if len(self.memory) < self.buffer_size:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.buffer_size

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action.squeeze(), reward.squeeze(), next_state, done.squeeze()

    def __len__(self):
        return len(self.memory)

