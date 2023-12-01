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
from gymnasium import Env
from gymnasium import spaces


npr = NumpyReader()




class CustomTensorDataReader(ABC):
    def __init__(self):

        # self.data = torch.tensor(npr.matrix.copy(), dtype=torch.float32).cpu().numpy()

        self.data = npr.matrix.copy()

        # self.data = self.data.cpu().numpy()

        self.target_data = torch.load('target.pt').cpu().numpy()
        self.index = torch.load('index.pt').cpu().numpy()
        self.starting_index = self.index.copy()
        self.base_index = torch.load('current_index.pt').cpu().numpy()



class Stepper(CustomTensorDataReader):
    def __init__(self):
        super().__init__()

        self.reset_step()

    def reset_step(self):
        self.current_step = 0
        self.ep_step = 0
        self.current_index = np.random.choice(self.base_index[:-30])


        



class Action(Stepper):
    def __init__(self):
        super().__init__()

        self.reset_step()

    def reset_action(self):
        self.action = 0
    
    def search_action(self, action_input):
        action = action_input

        if action == 1:
            self.action = 0
        if action == 2:
            self.action = 1
        elif action == 3:
            self.action = 2



    

        



class Window(Action):
    def __init__(self):
        super().__init__()

        self.reset_step()
        self.update_window()
        self.reset_action()

    def update_data_window(self):
        self.window = self.data[self.current_index]
        self.shape = self.window.shape

    def update_target_window(self):
        self.target_window = self.target_data[
            self.index[self.current_index][0] : self.index[self.current_index][1]
        ]
    
    def update_window(self):
        self.update_data_window()
        self.update_target_window()
    
    def reset_window(self):
        self.reset_step()
        return self.window, self.target_window
        # return self.window, {}


class Reward(Window):
    def __init__(self):
        super().__init__()
    
        self.reset_window()

    
    def reset_reward(self):
        self.reward = 0.0
    
    def calc_reward(self):
        target = self.target_window.sum()
        terminated = False
        #no red, agent did nothin
        if (target == 0) & (self.action == 0):
            self.reward += 0.05
        #no red, agent alerted someone
        elif (target == 0) & (self.action == 1):
            self.reward -= 0.10
        #no red, agent blocked url
        elif (target == 0) & (self.action == 2):
            self.reward -= 0.10
        #red, agent did nothing
        elif (target >= 1) & (self.action == 0):
            self.reward = -100
            terminated = True
        #red, agent alerted someone
        elif (target >= 1) & (self.action == 1):
            self.reward += 0.2
        #red, agent auto blocked url
        elif (target >= 1) & (self.action == 2):
            self.reward += 1.0
        
        elif (self.action == 0):
            self.reward = 0.0
        
        return terminated

class Simulator(Reward):
    def __init__(self):
        super().__init__()
        
    
    def reset(self):
        self.reset_step()
        self.reset_action()
        self.reset_window()
        self.reset_reward()
        self.current_step = 0

        observation = self.window
        info = {}
        return observation, info

    def little_step(self,action):
        self.search_action(action_input=action)
        terminated = self.calc_reward()
        reward = self.reward
        info = {
            'reward':reward
        }
        return reward, info, terminated
    
    def get_observation(self):
        self.update_window()
        observation = self.window
        info = {}
        return observation, info

    def get_next_state(self):
        self.current_step += 1
        self.current_index += 1
        self.update_window()
        observation = self.window
        info = {}
        return observation, info
    
    def step(self, action):
        truncated = False
        observation, info = self.get_observation()
        next_state, info = self.get_next_state()
        reward, info, terminated = self.little_step(action=action)

        if self.current_step >= 20:
            truncated = True 
        
        done = terminated or truncated

        if done:
            if (self.reward > 1.0) & (self.reward < 1.5):
                reward += 25
            
        return next_state, reward, terminated, truncated, info

    

        

class NetworkEnv(gym.Env):
     metadata = {'render.modes':['human']}

     def __init__(
             self,
     ):
         self.simulator = Simulator()
        #  self.action_space = spaces.Discrete(3, start=1)
         self.action_space = spaces.Discrete(3)
         self.observation_space = spaces.Box(low = -np.inf, 
                                             high = np.inf, 
                                             shape=self.simulator.shape, 
                                             dtype=np.float32)
     def step(self, action):
         observation, reward, terminated, truncated, info = self.simulator.step(action=action)
         return observation, reward, terminated, truncated, info 
     
     def reset(self):
         self.simulator.reset()
         return self.simulator.get_observation()









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


# class Reward(Window):
#     def __init__(self):
#         super().__init__()
#         self.reset_reward()

#     def reset_reward(self):
#         self.reward = 0
#         self.reward_mult = 0
    
#     def get_reward(self, input_action):
#         self.action = input_action
#         self.ratio = round((self.action / self.expenses), 2)
#         self.pr = round((self.action - self.expenses) / self.expenses, 2)



# class Simulator(Reward):
#     def __init__(self):
#         super().__init__()
#         self.reset_simulator()


#     def reset_simulator(self):
#         self.reset_step()
#         self.update_window()
#         self.reset_expenses()
#         self.reset_budget()
#         self.set_expenses()

#         self.reward = 0


        
    
#     def search_action(self, action_input):
#         action = action_input

#         if action == 0:
#             self.reward_mult = 0
#         elif action == 1:
#             self.reward_mult = 1
#         elif action == 2:
#             self.reward_mult = 2
    
#     def little_step(self, action):
#         self.search_action(action=action)
#         pass
            
            
#     def take_step(self):
#         pass









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




