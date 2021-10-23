import torch
import numpy as np
import random
import copy
from collections import deque
import cv2
from rlnn import Grid_DQN
import torch.nn as nn
import torch.optim as optim


class DQN_Agent:
    def __init__(self, env):

        self.env = env
        self.observation_space = env.observation_space #should be (210, 160, 3)
        self.action_space = env.action_space #Discrete(NOOP, FIRE, RIGHT, LEFT)

        self.replay_memory = list() # can't set capacity N to python list
        self.N = 2048#16384    #this is completely arbitrary and probably will need to be adjusted
        self.batch_size = 32 #may want to experiment with batch_size

        self.COUNTER_MAX = 16
        self.refresh_counter = self.COUNTER_MAX


        self.Q = Grid_DQN()
        
        self.frozen_Q = copy.deepcopy(self.Q)
        self.frozen_Q.eval()

        #epsilon used for e-greedy policy
        self.epsilon = .9
        #alhpa for learning rate
        self.alpha = .01 #we need to test this
        #gamma for discount factor
        self.gamma = .99

        #still prefer adamw but sgd since copying paper
        self.optimizer = optim.SGD(self.Q.parameters(), lr = self.alpha, weight_decay=0.01)
        self.criterion = nn.MSELoss()
        
        self.previous_state = None
        self.previous_action = None
    
    def load(self, filename):
        self.Q.load_state_dict(torch.load('./{}-weights.pth'.format(filename)))

    
    def save(self, filename):
        torch.save(self.Q.state_dict(), './{}-weights.pth'.format(filename))

    def step(self, observation):

        explore_or_exploit = random.random()

        observation = torch.tensor(observation, dtype=torch.float32)

        action = None

        if explore_or_exploit < self.epsilon:
            action = np.random.choice(self.action_space.n)
        else:
            with torch.no_grad():
                action = torch.argmax(self.Q(observation))
        self.previous_state = observation
        self.previous_action = action
        return action

    def observe(self, observation, reward, done):
        observation = torch.tensor(observation)
        reward = torch.tensor(reward)
        self.replay_memory.append((self.previous_state, self.previous_action, reward, observation))
        self.replay_memory = self.replay_memory[-self.N:] #deletes anything over N moves old
        self.replay(done)

    def replay(self, done):
        if len(self.replay_memory) < self.batch_size:
            return

        batch_indices = random.sample(range(len(self.replay_memory)), self.batch_size)
        for index in batch_indices:
            sample = self.replay_memory[index]
            state = sample[0]
            action = sample[1]
            reward = sample[2]
            next_state = sample[3]

            state = state.float()
            next_state = next_state.float()

            #I'm going to clear memory each new episode,
            #so I don't need to worry about sample being done in previous episode
            #Checking for finished episode and is most recent action in memory
            if done == True and index == len(self.replay_memory) -1:
                if(type(reward) != type(torch.zeros(1))):
                    target = torch.tensor(reward) # I think this was causing an error before tensor
                else:
                    target = reward
            else:
                target = reward + self.gamma * torch.max(self.frozen_Q(next_state))

            
            output = self.Q(state)
            output = output[action]

            #debugging, but I think i fixed it
            if type(output) != type(torch.zeros(1)):
                print(type(output))
                print('output: {}'.format(output))
                output = torch.tensor(output)
            if type(target) != type(torch.zeros(1)):
                print(type(target))
                print('target: {}'.format(target))
                target = torch.tensor(target)
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