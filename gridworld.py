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


		reward = -5.0
		#converting all values to float for uniformity with other environments
		if action == 0 or action == 2: #if down or right towards goal reward
			reward = 0.0

		if (self.agent_pos == self.goal_pos).all():
			reward = 100.0
		if (self.agent_pos == self.trap_pos).all():
			reward = -50.0
		if (self.agent_pos == self.sreward_pos).all():
			reward = 1.0
		if (self.agent_pos == self.lreward_pos).all():
			reward = 5.0


		if(sum(self.agent_pos) == 0):
			reward +=0.0
		elif(sum(self.agent_pos) == 1):
			reward +=1.0
		elif(sum(self.agent_pos) == 2):
			reward +=2.0
		elif(sum(self.agent_pos) == 3):
			reward +=3.0
		elif(sum(self.agent_pos) == 4):
			reward +=4.0
		elif(sum(self.agent_pos) == 5):
			reward +=5.0
		elif(sum(self.agent_pos) == 6):
			reward +=100.0


		# if(self.agent_pos == [0,0]):
		# 	reward +=1.0
		# elif(self.agent_pos == [1,0] or self.agent_pos == [0,1]):
		# 	reward +=2.0
		# elif(self.agent_pos == [2,0] or self.agent_pos == [1,1] or self.agent_pos == [0,2]):
		# 	reward +=3.0
		# elif(self.agent_pos == [3,0] or self.agent_pos == [2,1] or self.agent_pos == [1,2] or self.agent_pos == [0,3]):
		# 	reward +=4.0
		# elif(self.agent_pos == [3,1] or self.agent_pos == [2,2] or self.agent_pos == [1,3]):
		# 	reward +=5.0
		# elif(self.agent_pos == [3,2] or self.agent_pos == [2,3]):
		# 	reward +=6.0
		# elif(self.agent_pos == [3,3]):
		# 	reward +=100.0
		# else:
		# 	print('idk how but agent_pos isn\'t on board')
		# if (self.agent_pos == self.goal_pos).all():
		# 	reward = 100.0
		# if (self.agent_pos == self.trap_pos).all():
		# 	reward = -10.0
		# if (self.agent_pos == self.sreward_pos).all():
		# 	reward = 2.0
		# if (self.agent_pos == self.lreward_pos).all():
		# 	reward = 4.0

		self.timestep += 1
		done = True if (self.timestep >= self.max_timesteps) or (self.agent_pos == self.goal_pos).all() else False
		info = {}

		return observation, reward, done, info
		
	def render(self):
		plt.imshow(self.state)
		plt.show(block=False)
		plt.pause(.25)
		plt.close()

	def save_render(self, step):
		plt.imshow(self.state)
		fig = plt.gcf()
		fig.savefig('./images/step{}'.format(step))
		plt.show(block=False)
		plt.pause(.25)
		plt.close()