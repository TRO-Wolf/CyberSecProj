import pandas as pd
import numpy as np
import pyarrow.feather as feather
import zipfile
import os 

class FileDirectoryManager:
    def __init__(self):
        self.sub_dir = 'temp_directory'
        self.tem_file_name = 'temp.csv'
        self.full_path_name = './temp_directory/temp.csv'
        self.full_feather_name = './temp_directory/temp.feather'

    def check_directory(self):
        if not os.path.exists(self.sub_dir):
            os.mkdir(self.sub_dir)
    
    def check_file(self):
        '''check if the file exist'''
        if not os.path.exists(self.full_path_name):
            return False
        else:
            return True
            



'''
We are reading 3 zip files, then merging them into a single temporary csv file
Step 1 Check if the temp_directory exist, if not, make it
Step 2 Check if temp.csv exist in the temp_directory

'''
        

        


class ZipDataMerger(FileDirectoryManager):
    def __init__(self):
        super().__init__()
        print('make sure to add .gitignore to temp_directory')
        self.compressed_files_list = [
            'compressed_chunk_1.zip', 'compressed_chunk_2.zip', 'compressed_chunk_3.zip'
        ]
        self.base_files_list = [
            'chunk_1.csv', 'chunk_2.csv', 'chunk_3.csv'
        ]

        self.df_list = []
        

    

    '''Reads the 3 Zip files,then appends each csv file to a dataframe list'''
    def read_and_append(self):
        for compressed_file, base_file in zip(self.compressed_files_list, self.base_files_list):
            print(f'now reading {compressed_file}')
            with zipfile.ZipFile(compressed_file, 'r') as zipf:
                with zipf.open(base_file) as csv_file:
                    df = pd.read_csv(csv_file)
                    self.df_list.append(df)

    '''loops through dataframe list to create a single main dataframe'''
    def create_dataframe(self):
        self.df = pd.concat(i for i in self.df_list)



class FileMaker(ZipDataMerger):
    def __init__(self):
        super().__init__()

    def create_temporary_csv(self):
        self.df.to_csv(self.full_path_name)
        self.df.to_feather()

    
    def run(self):

        # if not os.path.exists(self.full_path_name):
        if not os.path.exists(self.full_feather_name):
            self.read_and_append()
            self.create_dataframe()
            self.df = self.df.reset_index()
            #self.df.to_csv(self.full_path_name)
            self.df.to_feather(self.full_feather_name)
        else:
            print('file already exist')
            return
        
        
    