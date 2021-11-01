

from gridworld import DeterministicEnvironment
from DQN_Grid_Agent import DQN_Agent

import matplotlib.pyplot as plt

MAX_TIMESTEPS = 20

x = []
y = []

env = DeterministicEnvironment()
agent = DQN_Agent(env)
#agent.load("1000easy-gridworld")
agent.load("working-easy-gridworld")

for i_episode in range(5):
    episode_reward = 0
    observation = env.reset()
    agent.epsilon = 0
    for t in range(MAX_TIMESTEPS):
        env.render()
        action = agent.step(observation)
        observation, reward, done, info = env.step(action)
        episode_reward += reward
        agent.observe(observation,reward,done)
        if done:
            x.append(i_episode)
            y.append(episode_reward)
            print("Reward: {}".format(episode_reward))
            break

env.close()

plt.plot(x,y)