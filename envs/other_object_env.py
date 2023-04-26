import numpy as np

class OtherObjectEnv:
    def __init__(self):
        self.state = None

    def reset(self):
        self.state = np.random.rand(3)
        return self.state

    def step(self, action):
        reward = self._get_reward(action)
        done = self._is_done()
        next_state = self._get_next_state()
        return next_state, reward, done, {}

    def _get_reward(self, action):
        # Calculate reward based on current state and action
        return np.random.rand()

    def _is_done(self):
        # Check if the episode is done based on current state
        return False

    def _get_next_state(self):
        # Calculate next state based on current state and action
        return np.random.rand(3)
