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
from collections import defaultdict
import re
from abc import ABC, abstractmethod




class TensorDataReader(ABC):
    def __init__(self):
        self.data = torch.load('main_tensor.pt')
        self.target_data = torch.load('target.pt')
        self.index = torch.load('index.pt')

