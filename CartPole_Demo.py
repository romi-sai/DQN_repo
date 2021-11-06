import gym
from Cartpole_Agent import CartPole_Agent

import matplotlib.pyplot as plt
import torch
import numpy as np

EPISODES = 200
MAX_TIMESTEPS = 1000

env = gym.make('CartPole-v1')
agent = CartPole_Agent(env)

agent.load('best-cartpole')

y_rewards = list()
y_average = list()
y_epsilon = list()
x_axis = list()
x_epsilon = list()

rewards = 0

average_reward = 0
for i_episode in range(EPISODES):
    agent.epsilon = 0 # no exploration in demo
    episode_reward = 0
    observation = env.reset()
    for t in range(MAX_TIMESTEPS):
        #env.render()
        # if i_episode > EPISODES-10:
        #     env.render()
        #action = env.action_space.sample()
        action = agent.step(observation)
        observation, reward, done, info = env.step(action)
        episode_reward += reward
        if done:
            average_reward = average_reward + (episode_reward - average_reward)/(i_episode+1)
            print("Episode {}: average: {} current reward: {}".format(i_episode, average_reward, episode_reward))
            y_rewards.append(episode_reward)
            y_average.append(average_reward)
            x_axis.append(i_episode)
            break
    y_epsilon.append(agent.epsilon)
    x_epsilon.append(i_episode)

#plt.title('Reward and Average Reward Per {} Episodes'.format(EPISODES/100))
plt.title('Reward and Cumulative Average Reward')
#plt.plot(x_axis,y_average, label='average reward')
plt.plot(x_axis,y_rewards, label='reward at episode')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.show()

plt.title('Epsilon Decay')
plt.plot(x_epsilon,y_epsilon)
plt.xlabel('Episode')
plt.ylabel('Epsilon')
#plt.legend()
plt.show()


agent.save("cartpole")
env.close()