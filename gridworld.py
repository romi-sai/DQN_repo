import gym
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt

class DeterministicEnvironment(gym.Env):
  metadata = { 'render.modes': [] }

  def __init__(self):
    self.observation_space = spaces.Discrete(16)
    self.action_space = spaces.Discrete(4)
    self.max_timesteps = 10

  def reset(self):
    self.timestep = 0
    self.agent_pos = [0,0] # rows, columns
    self.goal_pos = [3,3] # rows, columns
    self.trap_pos = [1,2]
    self.sreward_pos = [0,1]
    self.lreward_pos = [3,0]
    self.state = np.zeros((4,4))
    self.state[tuple(self.goal_pos)] = 0.2
    self.state[tuple(self.trap_pos)] = 0.3
    self.state[tuple(self.sreward_pos)] = 0.4
    self.state[tuple(self.lreward_pos)] = 0.5
    self.state[tuple(self.agent_pos)] = 1
    observation = self.agent_pos
    return observation

  def step(self, action):
    if action == 0: #down
      self.agent_pos[0] += 1
    if action == 1: #up
      self.agent_pos[0] -= 1
    if action == 2: #right
      self.agent_pos[1] += 1
    if action == 3: #left
      self.agent_pos[1] -= 1

    self.agent_pos = np.clip(self.agent_pos, 0, 3)
    self.state = np.zeros((4,4))
    self.state[tuple(self.goal_pos)] = 0.2
    self.state[tuple(self.trap_pos)] = 0.3
    self.state[tuple(self.sreward_pos)] = 0.4
    self.state[tuple(self.lreward_pos)] = 0.5
    self.state[tuple(self.agent_pos)] = 1
    observation = self.agent_pos


    #converting all values to float for uniformity with other environments
    reward = -5.0
    if (self.agent_pos == self.goal_pos).all():
      reward = 100.0
    if (self.agent_pos == self.trap_pos).all():
      reward = -50.0
    if (self.agent_pos == self.sreward_pos).all():
      reward = -1.0
    if (self.agent_pos == self.lreward_pos).all():
      reward = 1.0

    self.timestep += 1
    done = True if (self.timestep >= self.max_timesteps) or (self.agent_pos == self.goal_pos).all() else False
    info = {}

    return observation, reward, done, info
    
  def render(self):
    plt.imshow(self.state)
    plt.show(block=False)
    plt.pause(.5)
    plt.close()