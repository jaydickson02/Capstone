import argparse
from keras.models import load_model
from Environment import Environment
from DQN import DQNAgent
from DotAgent import dotAgent
import numpy as np
import os

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Load a specific model.')
parser.add_argument('--model', type=str, help='The model filename to load.')
args = parser.parse_args()

# Load the model from the specified file
model = load_model(args.model)

# Initialize the environment
Width = 1000
Height = 1000

dotAgent = dotAgent([Width/2, (Height/2)], [0, 0], 3, [0, 0, 0])

runtime = 1000
renderEnv = True

# Set the environment to render mode
env = Environment(dotAgent, runtime, renderEnv)

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
# Pass the loaded model to the DQNAgent constructor
agent = DQNAgent(state_size, action_size, model=model)

# Clear the terminal
os.system("clear")

for e in range(1000):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    done = False
    i = 0
    rewardTotal = 0
    while not done:
        action = agent.actGreedy(state)
        next_state, reward, done = env.next(action, state)
        next_state = np.reshape(next_state, [1, state_size])
        state = next_state
        i += 1
        rewardTotal += reward
    print(f"Agent survived for {i} ticks. Got a reward of {rewardTotal}")
