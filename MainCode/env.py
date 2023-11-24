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
from .DataReader import NumpyReader
import pyarrow.feather as feather
from collections import defaultdict,namedtuple, deque
import re
from abc import ABC, abstractmethod
import random

import gymnasium as gym


npr = NumpyReader()




class CustomTensorDataReader(ABC):
    def __init__(self):

        self.data = torch.tensor(npr.matrix.copy(), dtype=torch.float32)

        self.target_data = torch.load('target.pt').cpu().numpy()
        self.index = torch.load('index.pt').cpu().numpy()



    


        





class Stepper(CustomTensorDataReader):
    def __init__(self):
        super().__init__()

        self.reset_step()

    def reset_step(self):
        self.current_step = 0


class Window(Stepper):
    def __init__(self):
        super().__init__()

        self.update_window()

    def update_data_window(self):
        self.window = self.data[
                self.index[self.current_step][0] : self.index[self.current_step][1]
        ]

    def update_target_window(self):
        self.target_window = self.target_data[
            self.index[self.current_step][0] : self.index[self.current_step][1]
        ]
    
    def update_window(self):
        self.update_data_window()
        self.update_target_window()




class Expenses(Window):
    def __init__(self):
        super().__init__()
        self.reset_expenses()
    
    def reset_expenses(self):
        self.total_expenses = 1
        self.expenses = 0
    
    def set_expenses(self):
        self.expenses = self.window.shape[0]
        if self.target_window.sum() >= 1:
            self.expenses *= 2


class Budget(Expenses):
    def __init__(self):
        super().__init__()
        self.reset_budget()
        self.action_space = gym.spaces.Box(low=1, high=1553 * 2, dtype=int)

    def reset_budget(self):
        self.budget = 1
    
    def get_budget(self, input_action):
        self.set_expenses()
        # self.action = self.action_space.sample()[0]
        self.action = input_action 
        self.ratio = round((self.action / self.expenses), 2)
        self.pr = round((self.action - self.expenses) / self.expenses, 2)


class Reward(Window):
    def __init__(self):
        super().__init__()

        self.reset_reward()

    
    def reset_reward(self):
        self.reward = 0
    
    def get_reward(self, input_action):
        self.action = input_action
        self.ratio = round((self.action / self.expenses), 2)
        self.pr = round((self.action - self.expenses) / self.expenses, 2)



class Simulator(Reward):
    def __init__(self):
        super().__init__()

        self.reset_simulator()


    
    def reset_simulator(self):
        self.reset_step()
        self.update_window()
        self.reset_expenses()
        self.reset_budget()
        self.set_expenses()

        self.reward = 0
        # obs_tuple = (self.data[self.current_step], self.target_data[self.current_step], self.reward)

        return 
    
    def search_action(self, action_input):
        action = action_input

        if action == 0:
            pass
            
    def step(self):
        pass

        




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




