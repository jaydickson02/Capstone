import numpy as np
import tensorflow as tf
from time import time
from tqdm import tqdm
import os
from keras.callbacks import TensorBoard
from keras.models import load_model
from datetime import datetime

from DotAgent import dotAgent
from Environment import Environment
from DQN import DQNAgent


# Load a model
# model = load_model('model.h5')
model = None


# Replace with your dot agent implementation

runtime = 100
renderEnv = False
env = Environment(dotAgent, runtime, renderEnv)

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size, model)
batch_size = 32


# Get the current date and time
now = datetime.now()

# Format the date and time as a string
formatted_time = now.strftime("%Y-%m-%d_%H-%M-%S")

# Create the directory name
log_dir = "logs/{}".format(formatted_time)

# Create the TensorBoard writer
writer = tf.summary.create_file_writer(log_dir)

# Clear the terminal
os.system("clear")

# Step Counter
steps = 0

# Model Action Log Length tracker
model_action_log_length_tracker = 0

# Training loop
pbar = tqdm(range(0, 20000))
for e in pbar:
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    done = False
    total_reward = 0

    while not done:
        action = agent.act(state)
        next_state, reward, done, finishCondition = env.next(action)
        next_state = np.reshape(next_state, [1, state_size])

        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        with writer.as_default():
            tf.summary.scalar('Action', action, step=steps)

        if (model_action_log_length_tracker != len(agent.get_model_action_log())):
            with writer.as_default():
                tf.summary.scalar('Non Random Actions',
                                  agent.get_model_action_log()[-1], step=steps)

            model_action_log_length_tracker = len(agent.get_model_action_log())

        steps += 1

    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

    # Update the target network every 20 episodes
    if e % 20 == 0:
        agent.update_target_model()

    # Log the reward, loss, and epsilon for TensorBoard
    with writer.as_default():
        tf.summary.scalar('Reward', total_reward, step=e)
        tf.summary.scalar('Loss', agent.get_loss_log()[-1], step=e)  # Log the last loss
        tf.summary.scalar('Epsilon', agent.get_epsilon_log()[-1], step=e)  # Log the last epsilon
        tf.summary.text("Finish Condition", finishCondition, step=e) # Log the finish condition


    # Save the model every 10 episodes
    if e % 10 == 0:
        agent.save_model("model.h5")


    pbar.set_postfix({"Total Reward": total_reward, "loss": agent.get_loss_log(
    )[-1], "Epsilon": agent.get_epsilon_log()[-1]})
