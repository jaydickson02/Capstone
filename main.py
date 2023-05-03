from Planet import Planet
from Satellite import Satellite
from Environment import Environment
from DQN import DQN

import numpy as np
from collections import deque
from tqdm import tqdm

from numpy import asarray
from numpy import savetxt
from numpy import loadtxt

import os
import time


def train(agent, episodes, batch_size):
    # Train the agent

    # Set loaded values
    RewardListData = rewardList

    # Set best reward to 0
    bestReward = 0

    # Set start time
    start = time.time()

    # Initialise progress bar
    pbar = tqdm(range(episodes))

    for e in pbar:

        state = env.reset()
        state = np.reshape(state, [1, state_space_size])

        done = False
        rewardAmount = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.next(action)
            next_state = np.reshape(next_state, [1, state_space_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            rewardAmount += reward

        agent.replay(batch_size)

        if (rewardAmount > bestReward):
            bestReward = rewardAmount

        # Update values for tracking overall training progress
        RewardListData.append(rewardAmount)

        # Update progress bars best reward value
        pbar.set_postfix({'Best Reward': bestReward})

        # Save the reward list (List is a python list, not a numpy array)
        np.save("rewardList.npy", RewardListData)

    end = time.time()

    # Calculate total time
    totalTime = end-start
    totalTime = totalTime/60

    print(f"Training took {totalTime} Minutes.")


def test(agent, runtime):
    # Test the agent

    # Set the environment to render mode
    env = Environment(planet, satellite, G, runtime, True)

    while True:
        state = env.reset()
        state = np.reshape(state, [1, state_space_size])
        done = False
        i = 0
        rewardTotal = 0
        while not done:
            action = agent.actGreedy(state)
            next_state, reward, done = env.next(action)
            next_state = np.reshape(next_state, [1, state_space_size])
            state = next_state
            i += 1
            rewardTotal += reward
        print(f"Agent survived for {i} ticks. Got a reward of {rewardTotal}")


def evaluateTraining():

    # Output graphs
    import matplotlib.pyplot as plt

    # Load the reward list
    rewardList = np.load("rewardList.npy").tolist()

    # Graph the reward per episode

    # Figure 1
    plt.figure(1)
    plt.plot(rewardList)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward per Episode")

    if (len(rewardList) < 100):
        print("Not enough data to graph average reward per 100 episodes")

    else:
        # Graph the average reward per 100 episodes
        rewardListAverage = np.array(rewardList)
        rewardListAverage = rewardListAverage.reshape(-1, 100)
        rewardListAverage = np.average(rewardListAverage, axis=1)

        # Figure 2
        plt.figure(2)
        plt.plot(rewardList)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Average Reward per 100 Episodes")

    # Show the graphs
    plt.show()


# Run if main
if __name__ == "__main__":

    # Allow for arguments to be passed
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="Train the agent", action="store_true")
    parser.add_argument("--test", help="Test the agent", action="store_true")
    parser.add_argument(
        "--evaluate", help="Evaluate the training", action="store_true")

    # Pass batch size and episodes
    parser.add_argument("--batch_size", help="Batch size", type=int)
    parser.add_argument("--episodes", help="Number of episodes", type=int)

    # Pass runtime
    parser.add_argument("--runtime", help="Runtime of simulation", type=int)

    args = parser.parse_args()

    # If no arguments are passed, test the agent
    if not args.train and not args.test and not args.evaluate:
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

    # State and action space shape
    state_space_size = env.observation_space.shape[0]
    action_space_size = env.action_space.n

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

    # Load reward list
    try:
        rewardList = np.load("rewardList.npy").tolist()
        print(f"-- Loaded reward values")
    except:
        # Initialise Empty Array
        rewardList = []
        print("-- No reward values found")

    # Print completed episodes
    print(f"-- Model has completed {len(rewardList)} episodes" + "\n")

    # Train or test
    if args.train:
        print("Training...")
        train(agent, episodes, batch_size)

    if args.test:
        print("Testing...")
        test(agent, runtime)

    if args.evaluate:
        print("Evaluating...")
        evaluateTraining()
