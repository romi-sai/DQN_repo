
from gridworld import DeterministicEnvironment
from DQN_Agent import DQN_Agent

import matplotlib.pyplot as plt

EPISODES = 100
MAX_TIMESTEPS = 20

env = DeterministicEnvironment()
agent = DQN_Agent(env)
rewards = list()
epsilon_decay_rate = agent.epsilon / EPISODES

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
            if i_episode % 10 == 0:
                print("Episode {} reward: {}".format(i_episode, episode_reward))
                rewards.append(episode_reward)
            
            break
    agent.epsilon -= epsilon_decay_rate



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

agent.save("gridworld")
env.close()