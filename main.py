from Planet import Planet
from Satellite import Satellite
from Environment import Environment
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random

import time

G = 100
Width = 1000
Height = 1000  # Careful these are hardcoded in the environment for the reward function

# Create objects
planet = Planet([Width/2, Height/2], 20, 1, [0, 0, 0])
satellite = Satellite([Width/2, (Height/2) + 100], [0, 0], 3, [0, 0, 0])

# Create environment
env = Environment(planet, satellite, G, False)


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
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def remember(self, state, action, reward, next_state, done):
        memory.append((state, action, reward, next_state, done))


def train_dqn(agent, episodes, batch_size):

    scores = deque(maxlen=100)  # last 100 scores
    for e in tqdm(range(episodes)):
        state = env.reset()
        state = np.reshape(state, [1, 6])

        done = False
        i = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.next(action)
            next_state = np.reshape(next_state, [1, 6])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            i += 1
        scores.append(i)
        mean_score = np.mean(scores)
        if mean_score >= 1000:
            print(f"Environment solved in {e} episodes!")
            break
        # print(f"Episode {e} - Mean survival time over last 100 episodes was {mean_score} ticks.")
        agent.replay(batch_size)


learning_rate = 0.01
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
batch_size = 5
episodes = 500

memory = deque(maxlen=2000)

agent = DQN(env, learning_rate, gamma, epsilon, epsilon_decay, epsilon_min)

# Load Weights
agent.model.load_weights("model_weights.h5")
print("Loaded model weights")

start = time.time()
train_dqn(agent, episodes, batch_size)

end = time.time()

# Save Weights
agent.model.save_weights("model_weights.h5")
print("Saved model weights")

# Calculate total time
totalTime = end-start
totalTime = totalTime/60

print(f"Training took {totalTime} Minutes.")


print("")

input("Press Enter to test the agent...")

# Test the agent
env = Environment(planet, satellite, G, True)

while True:
    state = env.reset()
    state = np.reshape(state, [1, 6])
    done = False
    i = 0
    while not done:
        action = agent.act(state)
        next_state, reward, done = env.next(action)
        next_state = np.reshape(next_state, [1, 6])
        state = next_state
        i += 1
    print(f"Agent survived for {i} ticks.")
