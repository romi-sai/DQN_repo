import torch
import numpy as np
import random
import copy
from collections import deque
import cv2
from rlnn import DQN
import torch.nn as nn
import torch.optim as optim


class DQN_Agent:
    def __init__(self, env):

        self.env = env
        self.observation_space = env.observation_space #should be (210, 160, 3)
        self.action_space = env.action_space #Discrete(NOOP, FIRE, RIGHT, LEFT)

        self.replay_memory = list() # can't set capacity N to python list
        self.N = 512#16384    #this is completely arbitrary and probably will need to be adjusted
        self.batch_size = 128 #may want to experiment with batch_size

        self.COUNTER_MAX = 16
        self.refresh_counter = self.COUNTER_MAX

        self.current_state = deque()
        self.current_state.append(torch.zeros(84,84))
        self.current_state.append(torch.zeros(84,84))
        self.current_state.append(torch.zeros(84,84))
        self.current_state.append(torch.zeros(84,84))

        #Q should be a neural network that predicts Q-value based on image
        #Original input was 4x(210,160,3)
        #We will grayscale and resize to 4x(84,84)
        #Q will take input of (84,84,4)
        #PyTorch Convolution Layers have 4-D input
        #NCHW: N - batch_dim, C - channel_dim, H - height, W - width
        #Q has output layer of 4 for value of (NOOP, FIRE, RIGHT, LEFT)
        #We can select max of actions
        self.Q = DQN()
        #self.load('Breakout-v0')
        
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

        #preprocessing time
        #greyscale and downscaling (paper just cropped, but I think this is smarter)
        
        #(210,160,3) -> (210,160)
        observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        #(210, 160) -> (84,84)
        #I rendered and checked the image and the ball is still visible in Breakout
        #Breakout has the smallest part so all the others should work too
        #Otherwise I can make the image a little bigger or just crop
        observation = cv2.resize(observation, (84,84))

        #Gridworld render also is working correctly

        observation = torch.from_numpy(observation)

        explore_or_exploit = random.random()

        action = None

        if explore_or_exploit < self.epsilon:
            action = np.random.choice(self.action_space.n)
        else:
            self.current_state.popleft()
            self.current_state.append(observation)
            phi = torch.unsqueeze(torch.stack((self.current_state[0], self.current_state[1], self.current_state[2], self.current_state[3]),dim=0), dim=0)
            phi = phi.float()
            #I don't want to save the values of every iteration, but I'm pretty sure that
            #rerunning the DQN.forward() will add to the DAG of Functions.
            #I THINK that torch.no_grad() context manager will ensure that 
            # there's no "double dipping" of gradient computations
            with torch.no_grad():
                action = torch.argmax(self.Q(phi))
        self.previous_state = observation
        self.previous_action = action
        return action

    def observe(self, observation, reward, done):
        observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        observation = cv2.resize(observation, (84,84))
        observation = torch.from_numpy(observation)

        current_phi = torch.unsqueeze(torch.stack((self.current_state[0], self.current_state[1], self.current_state[2], self.current_state[3]), dim=0), dim=0)
        self.current_state.popleft()
        self.current_state.append(observation)
        next_phi = torch.unsqueeze(torch.stack((self.current_state[0], self.current_state[1], self.current_state[2], self.current_state[3]), dim=0), dim=0)
        self.replay_memory.append((current_phi, self.previous_action, reward, next_phi))
        self.replay_memory = self.replay_memory[-self.N:] #deletes anything over N moves old
        self.replay(done)

    def replay(self, done):
        if len(self.replay_memory) < self.batch_size:
            return

        batch_indices = random.sample(range(len(self.replay_memory)), self.batch_size)
        for index in batch_indices:
            sample = self.replay_memory[index]
            phi = sample[0]
            action = sample[1]
            reward = sample[2]
            next_phi = sample[3]

            phi = phi.float()
            next_phi = next_phi.float()

            #I'm going to clear memory each new episode,
            #so I don't need to worry about sample being done in previous episode
            #Checking for finished episode and is most recent action in memory
            if done == True and index == len(self.replay_memory) -1:
                target = torch.tensor(reward) # I think this was causing an error before tensor
            else:
                target = reward + self.gamma * torch.max(self.frozen_Q(next_phi))

            
            output = self.Q(phi)
            output = output[0,action]
            #print('output: {}'.format(type(output)))
            #print('output: {}'.format(output))
            #print('target: {}'.format(type(target)))
            #print('target: {}'.format(target))
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
        self.current_state.clear()
        self.current_state.append(torch.zeros(84,84))
        self.current_state.append(torch.zeros(84,84))
        self.current_state.append(torch.zeros(84,84))
        self.current_state.append(torch.zeros(84,84))