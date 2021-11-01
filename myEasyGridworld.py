from gridworld import DeterministicEnvironment
from DQN_Grid_Agent import DQN_Agent

import matplotlib.pyplot as plt
import torch
import numpy as np

EPISODES = 1000
MAX_TIMESTEPS = 64

env = DeterministicEnvironment()
agent = DQN_Agent(env)
y_rewards = list()
y_average = list()
y_epsilon = list()
x_axis = list()
x_epsilon = list()
epsilon_decay_rate = 0.995 #agent.epsilon / EPISODES


observation = env.reset()

average_reward = 0
for i_episode in range(EPISODES):
    episode_reward = 0
    observation = env.reset()
    for t in range(MAX_TIMESTEPS):
        #env.render()
        #action = env.action_space.sample()
        action = agent.step(observation)
        observation, reward, done, info = env.step(action)

        episode_reward += reward
        agent.observe(observation,reward,done)
        if done:
            if i_episode % (EPISODES / 100) == 0:
                average_reward = average_reward / (EPISODES/100)
                print("average episode {} reward: {}".format(i_episode, average_reward))
                y_rewards.append(episode_reward)
                y_average.append(average_reward)
                x_axis.append(i_episode)
                average_reward = 0
            else:
                average_reward += episode_reward
            break
    y_epsilon.append(agent.epsilon)
    x_epsilon.append(i_episode)
    agent.epsilon *= epsilon_decay_rate

plt.title('Reward and Average Reward Per {} Episodes'.format(EPISODES/100))
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
plt.legend()
plt.show()

q_values = np.zeros((4,4))
for row in range(4):
    for column in range(4):
        trow = torch.tensor(row/1.0)
        tcolumn = torch.tensor(column/1.0)
        q_values[row,column] = torch.max(agent.Q( torch.tensor([trow,tcolumn])))
plt.imshow(q_values)
plt.show()

x = []
y = []
agent.epsilon = 0

for i_episode in range(5):
    episode_reward = 0
    observation = env.reset()
    for t in range(MAX_TIMESTEPS):
        env.save_render(t)
        action = agent.step(observation)
        observation, reward, done, info = env.step(action)
        episode_reward += reward
        print("reward: {}".format(reward))
        if done:
            x.append(i_episode)
            y.append(episode_reward)
            print("Episode reward: {}".format(episode_reward))
            break

plt.title('Reward Greedily Following Optimal Policy')
plt.plot(x,y)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()


agent.save("easy-gridworld")
env.close()