import numpy as np
import pandas as pd
from DataManager import ZipDataMerger



# zip_reader = ZipDataMerger()
# zip_reader.read_data()
# zip_reader.create_dataframe()
# zip_reader.cre






class DataReader:
    def __init__(self):
        #Dataframe
        # self.base_data = pd.read_csv('Full_Data.csv')
        self.base_data = pd.read_csv('./temp_directory/temp.csv')




        self.base_data = self.base_data.rename({'Unnamed: 0.1':'Integer_index'},axis=1)
        # self.base_data = self.base_data.set_index('Integer_index')
        self.base_data = self.base_data.drop(['Unnamed: 0','test'],axis=1)
        self.base_data = self.base_data.set_index('Time Stamp')
        self.base_data = self.base_data.drop(['Unnamed: 0.2', 'Integer_index'],axis=1)

        #Index values for the time steps
        self.index_values = self.base_data.index.drop_duplicates()

        #Data in a numpy array for faster run time
        self.data = self.base_data.values

        self.list1 = []
        self.dummy_step = 0
        self.starting_value = 0
        step_size = self.base_data.loc[self.index_values[self.dummy_step]].__len__()

        self.starting_steps = 0


        for i in self.index_values[:-1]:
            window_start = self.starting_value
            window_end = self.starting_value + step_size
            window_size = (window_start, window_end)
            self.list1.append(window_size)

            self.dummy_step += 1
            self.starting_value += step_size
            step_size = self.base_data.loc[self.index_values[self.dummy_step]].__len__()
        

class TimeStep:

    def __init__(self):
        self.reader = DataReader()
        self.base_data = self.reader.base_data.copy()

        '''numpy array of our data'''
        self.data = self.reader.data

        self.index_array = self.reader.index_values

        self.index_list = self.reader.list1.copy()

        self.current_step = 0

    
    def array_step(self):
        current_tuple = self.index_list[self.current_step]
        observation = self.data[current_tuple[0] : current_tuple[1]]
        self.current_step += 1
        return observation

    
    def step(self):
        observation = self.base_data.loc[self.index_array[self.current_step]]
        self.current_step += 1
        return observation
