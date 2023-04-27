from Planet import Planet
from Satellite import Satellite
from Environment import Environment
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint

import random
import os
import time

G = 100
Width = 1000
Height = 1000  # Careful these are hardcoded in the environment for the reward function

# Create objects
planet = Planet([Width/2, Height/2], 20, 1, [0, 0, 0])
satellite = Satellite([Width/2, (Height/2) + 100], [0, 0], 3, [0, 0, 0])


class DQN:
    def __init__(self, env, learning_rate, gamma, epsilon, epsilon_decay, epsilon_min):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
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

    def replay(self, batch_size):
        if len(memory) < batch_size:
            return
        minibatch = random.sample(memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * \
                    np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target

            self.model.fit(state, target_f, epochs=1, verbose=0,
                           callbacks=[self.checkpoint])
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def remember(self, state, action, reward, next_state, done):
        memory.append((state, action, reward, next_state, done))


def train(agent, episodes, batch_size):
    # Train the agent

    EpisodeCount = completedEpisodes

    start = time.time()

    for e in tqdm(range(episodes)):
        state = env.reset()
        state = np.reshape(state, [1, 6])

        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.next(action)
            next_state = np.reshape(next_state, [1, 6])
            agent.remember(state, action, reward, next_state, done)
            state = next_state

        agent.replay(batch_size)
        EpisodeCount += 1

        # Save the value of completed episodes every time
        np.save("completedEpisodes.npy", EpisodeCount)

    end = time.time()

    # Calculate total time
    totalTime = end-start
    totalTime = totalTime/60

    print(f"Training took {totalTime} Minutes.")


def test(agent, runtime):
    # Test the agent
    env = Environment(planet, satellite, G, runtime, True)

    while True:
        state = env.reset()
        state = np.reshape(state, [1, 6])
        done = False
        i = 0
        reward = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.next(action)
            next_state = np.reshape(next_state, [1, 6])
            state = next_state
            i += 1
            reward += reward
        print(f"Agent survived for {i} ticks. Got a reward of {reward}")


# Run if main
if __name__ == "__main__":

    # Allow for arguments to be passed
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="Train the agent", action="store_true")
    parser.add_argument("--test", help="Test the agent", action="store_true")

    # Pass batch size and episodes
    parser.add_argument("--batch_size", help="Batch size", type=int)
    parser.add_argument("--episodes", help="Number of episodes", type=int)

    # Pass runtime
    parser.add_argument("--runtime", help="Runtime of simulation", type=int)

    args = parser.parse_args()

    # If no arguments are passed, test the agent
    if not args.train and not args.test:
        args.test = True

    # If batch size is passed, set it
    if args.batch_size:
        batch_size = args.batch_size
    else:
        batch_size = 10

    # If episodes is passed, set it
    if args.episodes:
        episodes = args.episodes
    else:
        episodes = 50

    # If runtime is passed, set it
    if args.runtime:
        runtime = args.runtime
    else:
        runtime = 1000

    # Hyperparameters
    learning_rate = 0.01
    gamma = 0.95
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01

    # Memory
    memory = deque(maxlen=2000)

    # Create environment
    env = Environment(planet, satellite, G, runtime, False)

    # Create agent
    agent = DQN(env, learning_rate, gamma, epsilon, epsilon_decay, epsilon_min)

    # Clear the screen
    os.system("clear")

    # Print hyperparameters
    print("Hyperparameters:")
    print(f"Learning Rate: {learning_rate}")
    print(f"Gamma: {gamma}")
    print(f"Epsilon: {epsilon}")
    print(f"Epsilon Decay: {epsilon_decay}")
    print(f"Epsilon Min: {epsilon_min}" + "\n")

    # Print arguments
    print(
        f"Runtime: {runtime}, Episodes: {episodes}, Batch Size: {batch_size}" + "\n")

    print("-- Initialised Environment")
    print("-- Created Agent" + "\n")

    # Load Weights
    try:
        agent.model.load_weights("best_model.h5")
        print("-- Loaded model weights")
    except:
        print("-- No model weights found")

    # Load completed episodes
    try:
        completedEpisodes = np.load("completedEpisodes.npy")
        print(f"-- Model has completed {completedEpisodes} episodes" + "\n")
    except:
        completedEpisodes = 0
        print("-- No episodes completed" + "\n")

    # Train or test
    if args.train:
        print("Training...")
        train(agent, episodes, batch_size)

    if args.test:
        print("Testing...")
        test(agent, runtime)
