
from gridworld import DeterministicEnvironment
from DQN_Grid_Agent import DQN_Agent

import matplotlib.pyplot as plt

EPISODES = 1000
MAX_TIMESTEPS = 20

env = DeterministicEnvironment()
agent = DQN_Agent(env)
rewards = list()
x_axis = list()
epsilon_decay_rate = 0.99 #agent.epsilon / EPISODES

observation = env.reset()

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
                print("Episode {} reward: {}".format(i_episode, episode_reward))
                rewards.append(episode_reward)
                x_axis.append(i_episode)
            break
    agent.epsilon *= epsilon_decay_rate



observation = env.reset()
for t in range(MAX_TIMESTEPS):
    env.render()
    action = agent.step(observation)
    observation, reward, done, info = env.step(action)
    print(reward)
    agent.observe(observation,reward,done)
    if done:
        print("Reward: {}".format(reward))
        break

agent.save("easy-gridworld")
env.close()