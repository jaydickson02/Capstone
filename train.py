import numpy as np
import tensorflow as tf
from time import time
from tqdm import tqdm
import os
from keras.callbacks import TensorBoard
from keras.models import load_model

from DotAgent import dotAgent
from Environment import Environment
from DQN import DQNAgent


# Initialize the environment
Width = 1000
Height = 1000

# Load a model
# model = load_model('model.h5')
model = None


# Replace with your dot agent implementation
dotAgent = dotAgent([Width/2, (Height/2)], [0, 0], 3, [0, 0, 0])
runtime = 1000
renderEnv = False
env = Environment(dotAgent, runtime, renderEnv)

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size, model)
batch_size = 32

# Create a TensorBoard writer
log_dir = "logs/{}".format(time())
writer = tf.summary.create_file_writer(log_dir)

# Clear the terminal
os.system("clear")

# Training loop
pbar = tqdm(range(6272, 20000))
for e in pbar:
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    done = False
    total_reward = 0

    while not done:
        action = agent.act(state)
        next_state, reward, done = env.next(action, state)
        next_state = np.reshape(next_state, [1, state_size])

        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

    # Update the target network every 4 episodes
    if e % 2 == 0:
        agent.update_target_model()

     # Log the reward, loss, and epsilon for TensorBoard
    if e % 4 == 0:
        with writer.as_default():
            tf.summary.scalar('Reward', total_reward, step=e)
            tf.summary.scalar('Loss', agent.get_loss_log()
                              [-1], step=e)  # Log the last loss
            tf.summary.scalar('Epsilon', agent.get_epsilon_log()
                              [-1], step=e)  # Log the last epsilon

    # Save the model every 10 episodes
    if e % 10 == 0:
        agent.save_model("model.h5")

    pbar.set_postfix({"Total Reward": total_reward, "loss": agent.get_loss_log(
    )[-1], "Epsilon": agent.get_epsilon_log()[-1]})
