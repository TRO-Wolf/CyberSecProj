import torch
import torch.utils.data as utils
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import torch.optim as optim


from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
from DataManager import FileMaker
import pyarrow.feather as feather
from collections import defaultdict,namedtuple, deque
import re
from abc import ABC, abstractmethod
import random

from Data import TensorDataset


class TensorDataReader(ABC):
    def __init__(self):
        self.data = torch.load('main_tensor.pt').cpu().numpy()
        self.target_data = torch.load('target.pt').cpu().numpy()
        self.index = torch.load('index.pt').cpu().numpy()



class Reward(TensorDataReader):
    def __init__(self):
        super().__init__()

        self.reset_reward()

    
    def reset_reward(self):
        self.reward = 0



class Simulator(TensorDataReader):
    def __init__(self):
        super().__init__()

        self.reset()


    
    def reset(self):
        self.current_step = 0
        self.reward = 0
        obs_tuple = (self.data[self.current_step], self.target_data[self.current_step], self.)

        return 
    
    def search_action(self, action_input):
        action = action_input

        if action == 0:
            


    def step(self, inp)




class ReplayMemory(object):


    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

        self.transition = namedtuple('Transition',
                                    ('state', 'action', 'next_state', 'reward'))

    def push(self, *args):
        """Save a transition"""
        self.memory.append(self.transition(*args))


    def sample(self, batch_size):
        return np.random.choice(self.memory, batch_size, replace=True)

    def __len__(self):
        return len(self.memory)
    

class DQNConv1D(nn.Module):
    def __init__(self, shape, actions_n):
        super(DQNConv1D, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(shape[0], 128, 5),
            nn.ReLU(),
            nn.Conv1d(128, 128, 5),
            nn.ReLU(),
        )

        out_size = self._get_conv_out(shape)

        self.fc_val = nn.Sequential(
            nn.Linear(out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.fc_adv = nn.Sequential(
            nn.Linear(out_size, 512),
            nn.ReLU(),
            nn.Linear(512, actions_n)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        val = self.fc_val(conv_out)
        adv = self.fc_adv(conv_out)
        return val + adv - adv.mean(dim=1, keepdim=True)




