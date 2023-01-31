import torch
from torch import nn, optim

class Q_net(nn.Module):
    def __init__(self, num_state, num_action):
        '''
        Class for making a 3 layer MLP for the Q network
        @ num_state : the number of states in the environment's observation space
        @ num_action : the number of possible actions in the environment's action space
        '''
        
        super().__init__()
    
        self.fc1 = nn.Linear(num_state, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, num_action)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x