import gym
from Cartpole_Agent import CartPole_Agent

import matplotlib.pyplot as plt
import torch
import numpy as np

EPISODES = 1000
MAX_TIMESTEPS = 1000

env = gym.make('CartPole-v1')
agent = CartPole_Agent(env)

y_rewards = list()
y_average = list()
y_epsilon = list()
x_axis = list()
x_epsilon = list()
#in general epsilon decay rate should be number of zeros long
#500 -> .99, 1000 -> .995, 5000 -> .999, 10000 -> .9995
epsilon_decay_rate = 0.995 #agent.epsilon / EPISODES

rewards = 0

average_reward = 0
for i_episode in range(EPISODES):
    episode_reward = 0
    observation = env.reset()
    for t in range(MAX_TIMESTEPS):
        #env.render()
        if i_episode > EPISODES-10:
            env.render()
        #action = env.action_space.sample()
        action = agent.step(observation)
        observation, reward, done, info = env.step(action)
        episode_reward += reward
        agent.observe(observation,reward,done)
        if done:
            average_reward = average_reward + (episode_reward - average_reward)/(i_episode+1)
            if i_episode % (EPISODES / 100) == 0:
                #average_reward = average_reward / (EPISODES/100)
                print("Episode {}: average: {} current reward: {}".format(i_episode, average_reward, episode_reward))
                y_rewards.append(episode_reward)
                y_average.append(average_reward)
                x_axis.append(i_episode)
                #average_reward = 0
            #else:
                #average_reward += episode_reward
            break
    agent.replay()
    y_epsilon.append(agent.epsilon)
    x_epsilon.append(i_episode)
    agent.epsilon *= epsilon_decay_rate

#plt.title('Reward and Average Reward Per {} Episodes'.format(EPISODES/100))
plt.title('Reward and Cumulative Average Reward')
plt.plot(x_axis,y_average, label='average reward')
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