import pandas as pd
import numpy as np
import zipfile
import os 

class FileDirectoryManager:
    def __init__(self):
        self.sub_dir = 'temp_directory'
        self.tem_file_name = 'temp.csv'
        self.full_path_name = './temp_directory/temp.csv'

    def check_directory(self):
        if not os.path.exists(self.sub_dir):
            os.mkdir(self.sub_dir)
    
    def check_file(self):
        '''check if the file exist'''
        if not os.path.exists(self.full_path_name):
            return False
        else:
            return True
            



        

        


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
        

    


    def read_data(self):


        for compressed_file, base_file in zip(self.compressed_files_list, self.base_files_list):
            with zipfile.ZipFile(compressed_file, 'r') as zipf:
                with zipf.open(base_file) as csv_file:
                    df = pd.read_csv(csv_file)
                    self.df_list.append(df)

    def create_dataframe(self):
        self.df = pd.concat(i for i in self.df_list)

    def create_temporary_csv(self):
        self.check_directory()

        self.df.to_csv(self.full_path_name)

        # self.df.to_csv('./temp_directory/temp.csv')
    
    def run(self):
        if self.check_file():
            self.read_data()
            self.create_dataframe()
            self.create_temporary_csv()
    