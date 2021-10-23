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
from DQN_Agent import DQN_Agent

EPISODES = 1
MAX_TIMESTEPS = 1000

env = gym.make('Breakout-v0')
agent = DQN_Agent(env)
rewards = 0
for i_episode in range(EPISODES):
    observation = env.reset()
    for t in range(MAX_TIMESTEPS):
        if t % 100 == 0:
            print(t)
        if(t > MAX_TIMESTEPS-100):
            env.render()
        #action = env.action_space.sample()
        action = agent.step(observation)
        observation, reward, done, info = env.step(action)
        rewards += reward
        agent.observe(observation,reward,done)
        #print(info)
        if done:
            #print("Episode finished after {} timesteps".format(t+1))
            #print("Observation: {}".format(observation))
            print("Reward: {}".format(reward))
            break

agent.save("breakout")
env.close()