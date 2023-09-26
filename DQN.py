from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import random
import os
from time import time
import contextlib
import io
from collections import deque


class DQNAgent:
    def __init__(self, state_size, action_size, model=None):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.gamma = 0.95    # Reward decay coefficient γ
        self.epsilon = 1  # exploration rate
        self.epsilon_min = 0.01
        self.epsilonStepCount = 10 # Number of steps to decay epsilon
        self.epsilon_decay = 0.01
        self.learning_rate = 0.01
        self.Nreplace = 2000  # Update target network every Nreplace step
        self.target_model = self._build_model()
        self.step = 0

        # If a model is provided, use it; otherwise, build the model
        if model:
            self.model = model
        else:
            self.model = self._build_model()

        self.update_target_model()

        # TensorBoard
        self.loss_log = []  # To store loss values
        self.epsilon_log = []  # To store epsilon values
        self.model_action_log = []  # To store action values

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(256, input_dim=self.state_size, activation='tanh'))
        model.add(Dense(256, activation='tanh'))
        model.add(Dense(128, activation='tanh'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(
            learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        # Copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)

        # Log the non random action taken by the network
        self.model_action_log.append(np.argmax(act_values[0]))

        return np.argmax(act_values[0])

    def actGreedy(self, state):
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * \
                    np.amax(self.target_model.predict(
                        next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target

            loss = self.model.train_on_batch(state, target_f)

            # Log the loss and epsilon
            self.loss_log.append(loss)
            self.epsilon_log.append(self.epsilon)

        # This runs once per episode
        self.step += 1

        if self.epsilon > self.epsilon_min and self.step % self.epsilonStepCount == 0:
            self.epsilon -= self.epsilon_decay

    def save_model(self, name):
        self.model.save(name, include_optimizer=True)

    def get_loss_log(self):
        return self.loss_log

    def get_epsilon_log(self):
        return self.epsilon_log

    def get_model_action_log(self):
        return self.model_action_log
