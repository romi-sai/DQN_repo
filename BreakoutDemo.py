import gym
import ale_py
from DQN_Agent import DQN_Agent

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

def demo():
    env = gym.make('Breakout-v0')
    agent = DQN_Agent(env)
    for i_episode in range(5):
        observation = env.reset()
        for t in range(1000):
            env.render()
            #action = env.action_space.sample()
            action = agent.step(observation)
            observation, reward, done, info = env.step(action)
            if done:
                break

demo()