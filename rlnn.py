import torch
import torch.nn as nn
import torch.nn.functional as F


#My DQN is pretty much as similar as the one in https://arxiv.org/pdf/1312.5602.pdf as possible
# This is okay though cuz assignment 2 says 
# "implement DQN from scratch following DeepMind's paper" :)

# The nature paper mentions 3 conv layers but only shows 2? I'll stick with two
# and see how well it works but may need modifying.
# expected: if agent fails to make good decisions based on vision then conv layers are issue

class Breakout_DQN(nn.Module):

    def __init__(self):
        super(Breakout_DQN, self).__init__()
        #The first paper performance didn't pan out too well so I switched to Nature one.
        #4 input image channel, 16 kernels/output channels, 8x8 kernels, stride 4
        #says to use rectifier nonlinearity which is relu
        #(seems weird to not say relu, paper might've been before relu was the common name)
        self.conv1 = nn.Conv2d(4, 16, 8, stride=4) #32 -> 16 simpler so I think this is okay
        #second hidden layer convolves 32 4x4 filters with stride 2, rect nonlinearity
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2) #64 -> 32 simpler so I think this is okay
        #third conovlution layer explained in nature
        #64 filters, 3x3, stride=1
        #self.conv3 = nn.Conv2d(32, 32, 3, stride=1) #64 -> 32 simpler, so I think this is okay
        #final hidden layer is fc and 256 rectifier units
        #84x84  (W - F + 2P)/S + 1
        #(84-8+0)/4 + 1 = 20
        #pool
        #(20-4+0)/2 + 1 = 9
        #(9-3+0)/1 + 1 = 7
        self.fc1 = nn.Linear(32 * 9 * 9, 128) #512 -> 128 simpler so I think this is okay

        #gridworld 4 actions: down, up, right, left
        #atari 4 actions: noop, fire, right, left
        #cartpole 2 actions: push left, push right
        #I don't want to remake any neural networks,
        #so I'm going to use the same network, different saved
        # weights and map outputs as the following:
        #0 - down, noop, push left
        #1 - up, fire, push left
        #2 - right, right, push right
        #3 - left, left, push right
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        #Usually pooling is used after convolution, but the output is already so small that
        #I think pooling would be detrimental to feature detection
        #I'll only do relu as the paper says.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        #x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1) #flatten conv -> fc
        x = F.relu(self.fc1(x))
        x = self.fc2(x) #we want q values for each
        return x


class Grid_DQN(nn.Module):

    def __init__(self):
        super(Grid_DQN, self).__init__()

        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CartPole_DQN(nn.Module):

    def __init__(self):
        super(CartPole_DQN, self).__init__()

        self.fc1 = nn.Linear(4, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MountainCar_DQN(nn.Module):

    def __init__(self):
        super(MountainCar_DQN, self).__init__()

        self.fc1 = nn.Linear(2, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LunarLander_DQN(nn.Module):

    def __init__(self):
        super(LunarLander_DQN, self).__init__()

        self.fc1 = nn.Linear(8, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x