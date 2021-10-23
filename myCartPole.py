import gym
from DQN_Agent import DQN_Agent


EPISODES = 1
MAX_TIMESTEPS = 100

env = gym.make('CartPole-v1')
agent = DQN_Agent(env)
rewards = 0

for i_episode in range(EPISODES):
    observation = env.reset()
    for t in range(MAX_TIMESTEPS):
        env.render()
        #action = env.action_space.sample()
        action = agent.step(observation)
        observation, reward, done, info = env.step(action)
        rewards += reward
        agent.observe(observation,reward,done)
        if done:
            print("Reward: {}".format(reward))
            break
agent.save("cartpole")
env.close()