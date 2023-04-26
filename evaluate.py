import torch
import numpy as np
from satellite_env import SatelliteEnv
from dqn_agent import DQNAgent

def evaluate(agent, env, num_episodes=10, max_steps=100):
    rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            episode_reward += reward
            if done:
                break
        rewards.append(episode_reward)
    avg_reward = np.mean(rewards)
    print("Average reward over {} evaluation episodes: {}".format(num_episodes, avg_reward))
    return avg_reward
