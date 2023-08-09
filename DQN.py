import random
import numpy as np
import os
import pickle
import math
from time import time
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.losses import Huber
from tensorflow.summary import create_file_writer, record_if
import contextlib
import io


class DQN:
    def __init__(self, env, learning_rate, gamma, epsilon, epsilon_decay, epsilon_min, memory, target_update_frequency):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_max = epsilon  # Maximum epsilon value
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory = memory
        self.action_space = self.env.action_space.n
        self.observation_space = self.env.observation_space.shape[0]
        self.model = self.build_model()
        self.target_model = self.build_model()  # Create target model
        self.update_target_model()  # Match the initial weights
        self.target_update_frequency = target_update_frequency
        self.target_update_counter = 0  # Add a counter to track target update steps
        self.checkpoint = ModelCheckpoint(
            "best_model.h5", monitor='loss', verbose=0, save_best_only=True, mode='auto', save_freq=1)

        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_dir = os.path.join("logs", f"DQN_{current_time}")
        self.tensorboard = TensorBoard(log_dir=log_dir)
        self.file_writer = create_file_writer(log_dir)

        self.tensorboard.set_model(self.model)

        self.step = 0  # Add a step counter

    def build_model(self):
        model = Sequential()
        model.add(Dense(128, input_dim=self.observation_space,
                  activation="relu", kernel_initializer=he_normal()))
        model.add(Dense(128, activation="relu",
                  kernel_initializer=he_normal()))
        model.add(Dense(128, activation="relu",
                  kernel_initializer=he_normal()))
        model.add(Dense(self.action_space, activation="linear",
                  kernel_initializer=he_normal()))
        model.compile(loss=Huber(delta=1.0), optimizer=Adam(
            learning_rate=self.learning_rate))

        return model

    def update_target_model(self):
        # Copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        self.epsilon = self.epsilon_min + \
            (self.epsilon_max - self.epsilon_min) * \
            math.exp(-self.epsilon_decay * self.step)  # Adjust epsilon
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
                # Use target_model to predict the future Q-values
                target = reward + self.gamma * \
                    np.amax(self.target_model.predict(
                        next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target

            with contextlib.redirect_stdout(io.StringIO()):
                history = self.model.fit(
                    state, target_f, epochs=1, verbose=1, callbacks=[self.checkpoint, self.tensorboard])

            loss = history.history['loss'][0]
            losses.append(loss)

        # After every replay, update the target model to match the current model
        self.update_target_model()

        losses = np.array(losses)
        Averageloss = np.mean(losses)

        # Log average loss and epsilon
        with self.file_writer.as_default(), record_if(lambda: True):
            tf.summary.scalar('Average Loss', Averageloss, step=self.step)
            tf.summary.scalar('Epsilon', self.epsilon, step=self.step)
            for layer in self.model.layers:
                for weight in layer.weights:
                    tf.summary.histogram(weight.name, weight, step=self.step)

        return (Averageloss, self.epsilon)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        # Increment the step counter here
        self.step += 1

        # Log reward
        with self.file_writer.as_default(), record_if(lambda: True):
            tf.summary.scalar('Reward', reward, step=self.step)

    def save_state(self, filename):
        # Save the entire model, including optimizer state
        self.model.save(filename + '_model')

        # Save other training state
        state = {
            'target_model_weights': self.target_model.get_weights(),
            'epsilon': self.epsilon,
            'memory': self.memory,
            'step': self.step
        }
        with open(filename + '_state.pkl', 'wb') as file:
            pickle.dump(state, file)

    def load_state(self, filename):
        # Load the entire model, including optimizer state
        self.model = tf.keras.models.load_model(filename + '_model')

        # Load other training state
        with open(filename + '_state.pkl', 'rb') as file:
            state = pickle.load(file)

        self.target_model.set_weights(state['target_model_weights'])
        self.epsilon = state['epsilon']
        self.memory = state['memory']
        self.step = state['step']
