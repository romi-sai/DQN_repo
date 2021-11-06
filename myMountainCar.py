import gym
from MountainCar_Agent import MountainCar_Agent

import matplotlib.pyplot as plt
import torch
import numpy as np

EPISODES = 5000
MAX_TIMESTEPS = 10000

env = gym.make('MountainCar-v0')
agent = MountainCar_Agent(env)

y_rewards = list()
y_average = list()
y_epsilon = list()
x_axis = list()
x_epsilon = list()
#in general epsilon decay rate should be number of zeros long
#500 -> .99, 1000 -> .995, 5000 -> .999, 10000 -> .9995
#epsilon_decay_rate = 0.999 #agent.epsilon / EPISODES
START_EPSILON = .9
END_EPSILON = 0
epsilon_decay_rate = (START_EPSILON-END_EPSILON)/EPISODES

rewards = 0
frames = 0
average_reward = 0
agent.epsilon = START_EPSILON
i_episode = 0
while(agent.epsilon > 0):
    i_episode += 1   
# for i_episode in range(EPISODES):
    episode_reward = 0
    observation = env.reset()
    for t in range(MAX_TIMESTEPS):
        frames+=1
        env.render()
        # if i_episode > EPISODES-2:
        #     env.render()
        if agent.epsilon < 3*epsilon_decay_rate:
            env.render()
        #action = env.action_space.sample()
        action = agent.step(observation)
        print(action)
        observation, reward, done, info = env.step(action)
        episode_reward += reward
        agent.observe(observation,reward,done)
        if done:
            average_reward = average_reward + (episode_reward - average_reward)/(i_episode+1)
            if i_episode % (EPISODES / 100) == 0:
                #average_reward = average_reward / (EPISODES/100)
                print("Frame: {} Episode {}: average: {:.3f} current reward: {} epsilon: {:.5f}".format(frames, i_episode, average_reward, episode_reward, agent.epsilon))
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
    #agent.epsilon *= epsilon_decay_rate
    if(episode_reward > average_reward):
        agent.epsilon -= epsilon_decay_rate
    #agent.epsilon -= epsilon_decay_rate

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