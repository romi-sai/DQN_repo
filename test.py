
import gym
import ale_py
from Breakout_Agent import DQN_Agent

import matplotlib.pyplot as plt
from time import sleep
import torch
import numpy as np

EPISODES = 5000
MAX_TIMESTEPS = 10000
REPLAY_START_SIZE = 50000

env = gym.make('Breakout-v0')
agent = DQN_Agent
average_reward = 0
for i_episode in range(EPISODES):
    episode_reward = 0
    observation = env.reset()
    for t in range(MAX_TIMESTEPS):
        action = np.random.choice(env.action_space.n)
        observation, reward, done, info = env.step(action)
        episode_reward += reward
        #print(info)
        if done:
            average_reward = average_reward + (episode_reward - average_reward)/(i_episode+1)
            if i_episode % (EPISODES / 1000) == 0:
                print("Episode {}: average: {:.5f} current reward: {}".format(i_episode, average_reward, episode_reward))
            break
