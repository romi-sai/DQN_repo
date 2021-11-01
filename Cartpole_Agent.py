import torch
import numpy as np
import random
import copy
from collections import deque
import cv2
from rlnn import CartPole_DQN
import torch.nn as nn
import torch.optim as optim


class CartPole_Agent:
    def __init__(self, env):

        self.env = env
        #obervation space:
        #Box(-4.8 to 4.8,     -Inf to Inf,    -.418 to .418 (24 deg), -Inf to Inf)
        #   (Cart Position,   Cart Velocity,  Pole Angle,             Pole Angular Velocity)
        self.observation_space = env.observation_space
        self.action_space = env.action_space #Discrete(Push Left, Push Right)

        self.replay_memory = list() # can't set capacity N to python list
        self.N = 8192#16384    #There appears to be a problem with catastrophic forgetting
        self.batch_size = 32 #may want to experiment with batch_size

        self.COUNTER_MAX = 16
        self.refresh_counter = self.COUNTER_MAX

        self.current_state = torch.zeros(4,1)

        #Q should be a neural network that predicts Q-value based on image
        #Input is [Position, velocity, angle, angular velocity]
        #Q has output layer of 2 for value of (LEFT, RIGHT)
        #We can select max of actions
        self.Q = CartPole_DQN()
        #self.load('Breakout-v0')
        
        self.frozen_Q = copy.deepcopy(self.Q)
        self.frozen_Q.eval()

        #epsilon used for e-greedy policy
        self.epsilon = 1
        #alhpa for learning rate
        self.alpha = .001 #we need to test this
        #gamma for discount factor
        self.gamma = .7

        #still prefer adamw but sgd since copying paper
        self.optimizer = optim.SGD(self.Q.parameters(), lr = self.alpha)#, weight_decay=0.01)
        self.criterion = nn.MSELoss()
        
        self.previous_state = None
        self.previous_action = None
    
    def load(self, filename):
        self.Q.load_state_dict(torch.load('./{}-weights.pth'.format(filename)))

    
    def save(self, filename):
        torch.save(self.Q.state_dict(), './{}-weights.pth'.format(filename))

    def step(self, observation):
        
        #Observation is numpy.ndarray
        observation = torch.from_numpy(observation)

        explore_or_exploit = random.random()

        action = None

        if explore_or_exploit < self.epsilon:
            action = np.random.choice(self.action_space.n)
        else:
            self.current_state = observation
            #I don't want to save the values of every iteration, but I'm pretty sure that
            #rerunning the DQN.forward() will add to the DAG of Functions.
            #I THINK that torch.no_grad() context manager will ensure that 
            # there's no "double dipping" of gradient computations
            with torch.no_grad():
                action = torch.argmax(self.Q(observation))
                action = action.int().item()
        self.previous_state = observation
        self.previous_action = action
        return action

    def observe(self, observation, reward, done):
        observation = torch.from_numpy(observation)

        reward = torch.tensor(reward)

        self.current_state = observation.float()

        self.replay_memory.append((self.previous_state, self.previous_action, reward, self.current_state, done))
        #self.replay_memory = self.replay_memory[-self.N:] #deletes anything over N moves old
        # Instead of deleting the oldest I will save the first 1/16
        #This is a test to prevent catastrophic forgetting
        if len(self.replay_memory) > self.N:
            del self.replay_memory[int(self.N/8)]
        #self.replay(done)

    def replay(self):
        if len(self.replay_memory) < self.batch_size:
            return

        batch_indices = random.sample(range(len(self.replay_memory)), self.batch_size)
        for index in batch_indices:
            sample = self.replay_memory[index]
            state = sample[0]
            action = sample[1]
            reward = sample[2]
            next_state = sample[3]
            done = sample[4]

            #I'm going to clear memory each new episode,
            #so I don't need to worry about sample being done in previous episode
            #Checking for finished episode and is most recent action in memory
            if done == True:
                target = reward
            else:
                target = reward + self.gamma * torch.max(self.frozen_Q(next_state))
            
            output = self.Q(state)
            output = output[action] #get the value of action taken

            loss = self.criterion(output, target) #loss will be based on action value and target
            self.optimizer.zero_grad()
            loss.backward()
            for parameter in self.Q.parameters():
                parameter.grad.data.clamp_(-1,1)
            self.optimizer.step()

            self.refresh_counter -= 1
            if self.refresh_counter <= 0:
                self.refresh_counter = self.COUNTER_MAX
                self.frozen_Q = copy.deepcopy(self.Q)

    def reset(self):
        self.replay_memory.clear()
        self.refresh_counter = self.COUNTER_MAX
        self.frozen_Q = copy.deepcopy(self.Q)
        self.current_state = torch.zeros(4,1)