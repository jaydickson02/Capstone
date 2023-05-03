from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import random
import numpy as np

import contextlib
import io


class DQN:
    def __init__(self, env, learning_rate, gamma, epsilon, epsilon_decay, epsilon_min, memory):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory = memory
        self.action_space = self.env.action_space.n
        self.observation_space = self.env.observation_space.shape[0]
        self.model = self.build_model()
        self.checkpoint = ModelCheckpoint(
            "best_model.h5", monitor='loss', verbose=0, save_best_only=True, mode='auto', save_freq=1)

    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.observation_space, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.action_space, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(
            learning_rate=self.learning_rate))
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_space)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def actGreedy(self, state):
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        losses = []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * \
                    np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target

            with contextlib.redirect_stdout(io.StringIO()):
                history = self.model.fit(
                    state, target_f, epochs=1, verbose=1, callbacks=[self.checkpoint])

            loss = history.history['loss'][0]

            losses.append(loss)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Calculate the average loss
        losses = np.array(losses)
        Averageloss = np.mean(losses)

        return (Averageloss, self.epsilon)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
