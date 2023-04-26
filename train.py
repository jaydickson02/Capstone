import torch
import numpy as np
from satellite_env import SatelliteEnv
from dqn_agent import DQNAgent
from evaluate import evaluate

def train(agent, env, num_episodes=1000, max_steps=100, batch_size=64, 
          gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, 
          target_update=10, checkpoint_interval=100, checkpoint_path=None):
    epsilon = epsilon_start
    rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        for step in range(max_steps):
            action = agent.act(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done, batch_size, gamma)
            state = next_state
            episode_reward += reward
            if done:
                break
        rewards.append(episode_reward)
        epsilon = max(epsilon_end, epsilon_decay * epsilon)
        if episode % target_update == 0:
            agent.update_target_network()
        if episode % checkpoint_interval == 0 and checkpoint_path is not None:
            torch.save(agent.q_network.state_dict(), checkpoint_path)
        if episode % 100 == 0:
            print("Episode {}: Average reward over last 100 episodes: {}".format(
                episode, np.mean(rewards[-100:])))
        if np.mean(rewards[-100:]) >= 195.0:
            print("Environment solved in {} episodes!".format(episode - 99))
            break
    return rewards
