import numpy as np
import matplotlib.pyplot as plt

def plot_rewards(rewards, window=100):
    # Plot the rewards obtained by the agent over time
    smoothed_rewards = np.convolve(rewards, np.ones(window) / window, mode='valid')
    plt.plot(range(len(smoothed_rewards)), smoothed_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Smoothed Reward (Over Last {} Episodes)".format(window))
    plt.show()
