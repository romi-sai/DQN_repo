
#print('gym:', gym.__version__)
#print('ale_py:', ale_py.__version__)
#env = gym.make('ALE/Breakout-v5', render_mode='human')
'''
Useful calls:

from pprint import pprint
pprint(vars(env)):

{'_action_space': None,
 '_elapsed_steps': 0,
 '_max_episode_steps': 10000,
 '_metadata': None,
 '_observation_space': None,
 '_reward_range': None,
 'env': <gym.envs.atari.environment.AtariEnv object at 0x10a0b4f40>}

env.unwrapped.get_action_meanings()
env.action_space
env.observation_space.shape
'''


import gym
import ale_py
from Breakout_Agent import DQN_Agent

import matplotlib.pyplot as plt
from time import sleep
import torch

EPISODES = 5
MAX_TIMESTEPS = 10000
REPLAY_START_SIZE = 50000

env = gym.make('Breakout-v0')
agent = DQN_Agent(env)
agent.load('breakout')

y_rewards = list()
y_average = list()
y_epsilon = list()
x_axis = list()
x_epsilon = list()

# agent.debug = True
average_reward = 0
for i_episode in range(EPISODES):
    episode_reward = 0
    observation = env.reset()
    for t in range(MAX_TIMESTEPS):
        if t % 100 == 0:
            print('----------------------{}------------------'.format(t))
            agent.debug = True
        else:
            agent.debug = False
        env.render()
        action = agent.step(observation)
        sleep(.01)
        print('t {}: {}'.format(t,action))#Discrete(NOOP, FIRE, RIGHT, LEFT)
        observation, reward, done, info = env.step(action)
        episode_reward += reward
        #print(info)
        if done:
            average_reward = average_reward + (episode_reward - average_reward)/(i_episode+1)
            if i_episode % (EPISODES / 100) == 0:
                print("Episode {}: average: {} current reward: {}".format(i_episode, average_reward, episode_reward))
                y_rewards.append(episode_reward)
                y_average.append(average_reward)
                x_axis.append(i_episode)
            break
    y_epsilon.append(agent.epsilon)
    x_epsilon.append(i_episode)

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