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

#approx 100,000 frames per 400 episode
#aka 250 frames per episode
#this will probably increase with improved episode
#40,000 * 250 =  10,000,000
#4,000 * 250 = 1,000,000
EPISODES = 4000
MAX_TIMESTEPS = 10000
REPLAY_START_SIZE = 50000

env = gym.make('Breakout-v0')
agent = DQN_Agent(env)

y_rewards = list()
y_average = list()
y_epsilon = list()
x_axis = list()
x_epsilon = list()
#in general epsilon decay rate should be number of zeros long
#100 -> .95, 200 -> .97, 500 -> .99, 1000 -> .995, 5000 -> .999, 10000 -> .9995
#epsilon_decay_rate = 0.99 #agent.epsilon / EPISODES
#Paper does linear decay
STARTING_E = 1
FINAL_E = 0 #paper does to .1, but I don't like that as much
epsilon_decay_rate = (STARTING_E - FINAL_E)/EPISODES

#initial 50,000 moves for training purposes
while len(agent.replay_memory) < REPLAY_START_SIZE:
    observation = env.reset()
    for t in range(MAX_TIMESTEPS):
        if len(agent.replay_memory) % 5000 == 0:
            print("Mem: {}".format(len(agent.replay_memory)))
        action = agent.step(observation)
        observation, reward, done, info = env.step(action)
        agent.observe(observation,reward,done) #this will add to replay_mem without training
        if done:
            break
print("Initializing Replay Memory Finished")

# agent.debug = True
frames = 0
average_reward = 0
for i_episode in range(EPISODES):
    episode_reward = 0
    observation = env.reset()
    for t in range(MAX_TIMESTEPS):
        # if t % 100 == 0:
        #     print('----------------------{}------------------'.format(t))
        #     agent.debug = True
        # else:
        #     agent.debug = False
        #env.render()
        frames += 1
        action = agent.step(observation)
        if i_episode > EPISODES-2:
            env.render()
            sleep(.005)
            print('t {}: {}'.format(t,action))#Discrete(NOOP, FIRE, RIGHT, LEFT)
        observation, reward, done, info = env.step(action)
        episode_reward += reward
        agent.observe(observation,reward,done)
        #print(info)
        if done:
            average_reward = average_reward + (episode_reward - average_reward)/(i_episode+1)
            if i_episode % (EPISODES / 1000) == 0:
                print("Frame: {} Episode {}: average: {:.5f} current reward: {} epsilon: {:.5f}".format(frames, i_episode, average_reward, episode_reward, agent.epsilon))
                y_rewards.append(episode_reward)
                y_average.append(average_reward)
                x_axis.append(i_episode)
                agent.save("breakout-{}".format(EPISODES))
            break
        agent.replay()
    y_epsilon.append(agent.epsilon)
    x_epsilon.append(i_episode)
    #agent.epsilon *= epsilon_decay_rate
    agent.epsilon -= epsilon_decay_rate

#The plotting can be glitchy and slow, so I want to ensure model gets saved first
agent.save("breakout")

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

#agent.save("breakout")
env.close()