from Planet import Planet
from Satellite import Satellite
from dotAgent import dotAgent
from Environment import Environment
from DQN import DQN

import numpy as np
from collections import deque
from tqdm import tqdm

import os
import time

# Allow for arguments to be passed
import argparse


def investigate(list):
    # Investigate a list of values

    # Seperate into individual lists
    Action = [x[0] for x in list]
    Next = [x[1] for x in list]
    Reshape = [x[2] for x in list]
    Remember = [x[3] for x in list]
    StoreNextState = [x[4] for x in list]
    IterateReward = [x[5] for x in list]

    # Investigate each list
    print("Action: " + str(np.mean(Action)))
    print("Next: " + str(np.mean(Next)))
    print("Reshape: " + str(np.mean(Reshape)))
    print("Remember: " + str(np.mean(Remember)))
    print("StoreNextState: " + str(np.mean(StoreNextState)))
    print("IterateReward: " + str(np.mean(IterateReward)))


def train(agent, episodes, batch_size, verbose=0):
    # Train the agent

    TrainingDataList = TrainingData

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

        ReplayTrigger = 100
        ReplayTracker = 0

        loss = 0
        newEpsilon = 0

        while not done:

            action = agent.act(state)
            next_state, reward, done = env.next(action, state)
            next_state = np.reshape(next_state, [1, state_space_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            rewardAmount += reward

            if (ReplayTracker == ReplayTrigger):
                if (len(agent.memory) < batch_size):
                    batch_size = len(agent.memory)

                loss, newEpsilon = agent.replay(batch_size)
                ReplayTracker = 0

            ReplayTracker += 1

        if (rewardAmount > bestReward):
            bestReward = rewardAmount

        agent.save_state('saved_state.pkl')

        # Update values for tracking overall training progress
        TrainingDataList.append([rewardAmount, loss, newEpsilon])

        np.save("TrainingData.npy", np.array(TrainingDataList, dtype=object))

        if verbose == 0:
            pbar.set_postfix({'Best Reward': bestReward})

        if verbose == 1:
            pbar.set_postfix({'Best Reward': bestReward,
                             'Current Reward': rewardAmount, 'Loss': loss, 'Epsilon': newEpsilon})

    end = time.time()

    # Calculate total time
    totalTime = end-start
    totalTime = totalTime/60

    print(f"Training took {totalTime} Minutes.")


def test(agent, runtime):
    # Test the agent

    # Set the environment to render mode
    env = Environment(DotAgent, planet, satellite, G, runtime, True)

    while True:
        state = env.reset()
        state = np.reshape(state, [1, state_space_size])
        done = False
        i = 0
        rewardTotal = 0
        while not done:
            action = agent.actGreedy(state)
            next_state, reward, done = env.next(action, state)
            next_state = np.reshape(next_state, [1, state_space_size])
            state = next_state
            i += 1
            rewardTotal += reward
        print(f"Agent survived for {i} ticks. Got a reward of {rewardTotal}")


def evaluateTraining(combine):

    # Output graphs
    import matplotlib.pyplot as plt

    # Load the reward list
    ValueList = np.load("TrainingData.npy", None, True).tolist()

    # Graph the reward per episode
    xValues = np.arange(len(ValueList))

    # Filter list data
    rewardList = [x[0] for x in ValueList]

    lossList = [x[1] for x in ValueList]

    epsilonList = [x[2] for x in ValueList]

    # Add trendline
    z = np.polyfit(xValues, rewardList, 1)
    p = np.poly1d(z)

    if (combine == True):

        # Combine the graphs into one
        plt.figure(1)
        plt.plot(xValues, rewardList)
        plt.plot(xValues, p(xValues), "r--")
        plt.plot(xValues, lossList)
        plt.plot(xValues, epsilonList)
        plt.xlabel("Episode")
        plt.ylabel("Reward/Loss/Epsilon")
        plt.title("Reward/Loss/Epsilon per Episode")

    else:
        # Figure 1
        plt.figure(1)
        plt.plot(xValues, rewardList)
        plt.plot(xValues, p(xValues), "r--")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Reward per Episode")

        # Figure 2
        plt.figure(2)
        plt.plot(xValues, lossList)
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.title("Loss per Episode")

        # Figure 3
        plt.figure(3)
        plt.plot(xValues, epsilonList)
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")
        plt.title("Epsilon per Episode")

    # Show the graphs
    plt.show()


# Run if main
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # General actions
    parser.add_argument("--train", help="Train the agent", action="store_true")
    parser.add_argument("--test", help="Test the agent", action="store_true")
    parser.add_argument(
        "--evaluate", help="Evaluate the training", action="store_true")

    # Hyperparameters
    parser.add_argument("--batch_size", help="Batch size",
                        type=int, default=10)
    parser.add_argument(
        "--episodes", help="Number of episodes", type=int, default=50)
    parser.add_argument(
        "--runtime", help="Runtime of simulation", type=int, default=1000)
    parser.add_argument("--verbose", help="Verbose level", type=int, default=0)
    parser.add_argument("--combine", help="Combine graphs",
                        action="store_true", default=False)

    # Special features
    parser.add_argument("--genetic", help="Genetic Algorithm",
                        action="store_true", default=False)

    args = parser.parse_args()

    # If no arguments are parsed, test the agent
    args.test = args.test or not args.train and not args.evaluate

    # Set values based on parsed arguments or use defaults
    batch_size = args.batch_size or 10
    episodes = args.episodes or 50
    runtime = args.runtime or 1000
    verbose = args.verbose or 0
    combine = args.combine or False

    # Evaluate training and skip the rest
    if args.evaluate:
        # Clear the screen
        os.system("clear")
        print("Evaluating...")
        evaluateTraining(combine)
        exit()

    # Load Training Data list
    try:
        TrainingData = np.load("TrainingData.npy", None, True).tolist()
        TLoad = True
    except:
        # Initialise Empty list
        TrainingData = []
        TLoad = False

    # Hyperparameters
    learning_rate = 0.001
    gamma = 0.99
    epsilon_decay = 0.00001
    epsilon_min = 0.01
    target_update_frequency = 25

    # Find the last entry in the training data
    if TLoad:
        lastEntry = TrainingData[-1]
        epsilon = lastEntry[2]
    else:
        epsilon = 1

    # Environment Parameters
    G = 100
    Width = 1000
    Height = 1000  # Careful these are hardcoded in the environment for the reward function

    # Create objects
    planet = Planet([Width/2, Height/2], 20, 1, [0, 0, 0])
    satellite = Satellite([Width/2, (Height/2)], [0, 0], 3, [0, 0, 0])
    DotAgent = dotAgent([Width/2, (Height/2)], [0, 0], 3, [0, 0, 0])

    # Create environment
    env = Environment(DotAgent, planet, satellite, G, runtime, False)
    env.reset()

    # State and action space shape
    state_space_size = env.observation_space.shape[0]
    action_space_size = env.action_space.n

    # Memory
    memory = deque(maxlen=10000)

    # Create agent
    agent = DQN(env, learning_rate, gamma, epsilon,
                epsilon_decay, epsilon_min, memory, target_update_frequency)

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

    # Print environment and agent info
    print("-- Initialised Environment")
    print("-- Created Agent" + "\n")

    # Print training data info
    if TLoad:
        print("-- Loaded Training Data")
        print(f"-- Training Data has {len(TrainingData)} entries" + "\n")
    else:
        print("-- No Training Data Found" + "\n")

    # Load Weights
    try:
        agent.load_state('saved_state.pkl')
        print("-- Loaded model weights")
    except:
        print("-- No model weights found")

    # Print completed episodes
    print(
        f"-- Model has completed {len(TrainingData)} episodes of training" + "\n")

    # Train or test depending on arguments
    if args.train:
        print("Training..." + "\n")
        train(agent, episodes, batch_size, verbose)

    if args.test:
        print("Testing...")
        test(agent, runtime)
