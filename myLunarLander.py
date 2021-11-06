import gym
from LunarLander_Agent import LunarLander_Agent

import matplotlib.pyplot as plt
import torch
import numpy as np


EPISODES = 1000
MAX_TIMESTEPS = 10000

env = gym.make('LunarLander-v2')
agent = LunarLander_Agent(env)
# print(env.action_space)
# print(env.observation_space)

y_rewards = list()
y_average = list()
y_epsilon = list()
x_axis = list()
x_epsilon = list()
#in general epsilon decay rate should be number of zeros long
#500 -> .99, 1000 -> .995, 5000 -> .999, 10000 -> .9995
#epsilon_decay_rate = 0.999 #agent.epsilon / EPISODES
START_EPSILON = 1
END_EPSILON = 0
epsilon_decay_rate = (START_EPSILON-END_EPSILON)/EPISODES


rewards = 0
frames = 0
average_reward = 0
agent.epsilon = START_EPSILON
i_episode = 0
for i_episode in range(EPISODES):
    episode_reward = 0
    observation = env.reset()
    for t in range(MAX_TIMESTEPS):
        frames+=1
        if agent.epsilon < 2*epsilon_decay_rate:
            env.render()
        action = agent.step(observation)
        observation, reward, done, info = env.step(action)
        episode_reward += reward
        agent.observe(observation,reward,done)
        agent.replay()
        if done:
            average_reward = average_reward + (episode_reward - average_reward)/(i_episode+1)
            if i_episode % (EPISODES / 100) == 0:
                print("Frame: {} Episode {}: average: {:.3f} current reward: {:.3f} epsilon: {:.5f}".format(frames, i_episode, average_reward, episode_reward, agent.epsilon))
                agent.save("lunarlander-checkpoint")
                y_rewards.append(episode_reward)
                y_average.append(average_reward)
                x_axis.append(i_episode)
            break
    #agent.replay()
    y_epsilon.append(agent.epsilon)
    x_epsilon.append(i_episode)
    #agent.epsilon *= epsilon_decay_rate
    agent.epsilon -= epsilon_decay_rate

agent.save("lunarlander")

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


env.close()