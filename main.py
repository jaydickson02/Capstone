from Planet import Planet
from Satellite import Satellite
from Environment import Environment
from DQN import DQN

import numpy as np
from collections import deque
from tqdm import tqdm

import os
import time


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
            action = agent.actGreedy(state)
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

    # Environment Parameters
    G = 100
    Width = 1000
    Height = 1000  # Careful these are hardcoded in the environment for the reward function

    # Create objects
    planet = Planet([Width/2, Height/2], 20, 1, [0, 0, 0])
    satellite = Satellite([Width/2, (Height/2)], [0, 0], 3, [0, 0, 0])

    # Create environment
    env = Environment(planet, satellite, G, runtime, False)
    env.reset()

    # Memory
    memory = deque(maxlen=2000)

    # Create agent
    agent = DQN(env, learning_rate, gamma, epsilon,
                epsilon_decay, epsilon_min, memory)

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
