import numpy as np
import pandas as pd
from .FileManager import FileMaker
import pyarrow.feather as feather
from collections import defaultdict
import re

from abc import ABC, abstractmethod

FileMaker().run()

class NumpyReader(ABC):
    def __init__(self):
        self.main_data_list = ['main_matrix1.npz', 'main_matrix2.npz', 'main_matrix3.npz']
        matrix_list = []
        for i in self.main_data_list:
            matrix = np.load(i)
            matrix = list(matrix.values())
            matrix_list.append(matrix[0])
        self.matrix = np.concatenate(matrix_list, axis=0)
        del matrix_list





class BaseDataReader:
    def __init__(self):
        self.base_data = pd.read_feather('./temp_directory/temp.feather')

        self.base_data = self.base_data.drop([
            'index', 'Unnamed: 0.1','Unnamed: 0','test',
        ],axis=1)
        self.base_data = self.base_data.set_index('Time Stamp')


class TokenManager(BaseDataReader):
    def __init__(self):
        super().__init__()

class SimpleTokenizer:
    def __init__(self):
        self.token_to_id = defaultdict(lambda: len(self.token_to_id))
        self.token_to_id['<PAD>'] = 0  # Padding token

    def tokenize(self, text):
        # Simple tokenization by splitting on non-word characters
        tokens = re.findall(r'\w+|\S', text)
        return tokens

    def convert_tokens_to_ids(self, tokens):
        return [self.token_to_id[token] for token in tokens]

    def tokenize_column(self, series):
        # Apply tokenization to each row in the pandas series
        tokenized = series.apply(self.tokenize)
        return tokenized
    
    def run_convert(self):
        pass







class DataMod(BaseDataReader):
    def __init__(self):
        super().__init__()

        self.tokenizer = SimpleTokenizer()




        '''
        creates an array of values that include each
        seconds time stamp as an integer, drops all duplicates
        '''
        self.index_values = self.base_data.index.drop_duplicates()
        



        #Data in a numpy array for faster run time
        self.data = self.base_data.values


        self.window_tuples = []
        self.dummy_step = 0
        self.starting_value = 0

        '''
        step size is the number of logins are a specific time stamp
        '''
        step_size = self.base_data.loc[self.index_values[self.dummy_step]].__len__()

        self.starting_steps = 0


        for i in self.index_values[:-1]:
            window_start = self.starting_value
            window_end = self.starting_value + step_size
            window_size = (window_start, window_end)
            self.window_tuples.append(window_size)

            self.dummy_step += 1
            self.starting_value += step_size
            step_size = self.base_data.loc[self.index_values[self.dummy_step]].__len__()

class TimeStep:

    def __init__(self):
        self.reader = DataMod()

        '''Dataframe'''
        self.base_data = self.reader.base_data.copy()
        # self.base_data = self.base_data.drop(['target'],axis=1)

        '''numpy array of our data'''
        # self.data = self.reader.data
        self.data = self.base_data.values

        self.index_array = self.reader.index_values

        self.index_list = self.reader.window_tuples.copy()


        self.current_step = 0

        self.main_matrix = np.zeros((1553, 8))

    def reset(self):
        self.current_step = 0

    
    def array_step(self):
        current_tuple = self.index_list[self.current_step]
        current_window = self.index_list[self.current_step]
        observation = self.data[current_window[0] : current_window[1]]
        self.current_step += 1
        return observation

    
    def step(self):
        observation = self.base_data.loc[self.index_array[self.current_step]]
        self.current_step += 1
        return observation
    


