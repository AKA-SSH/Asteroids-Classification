import os
import sys

import pandas as pd

from utils.logger import logging
from utils.exception import CustomException

from utils.pickle_file import pickle_file
from utils.unpickle_file import unpickle_file

class DataIngestion:
    def __init__(self):
        pass
    
    def data_ingestion(self, raw_data_file_path= r'Data\raw_data.csv'):
        try:
            logging.info('data ingestion initiated')
            
            logging.info(f'loading raw data for {raw_data_file_path}')
            raw_dataframe= pd.read_csv(raw_data_file_path, low_memory= False)
            selected_columns= ['orbit_id', 'epoch', 'e', 'i', 'om', 'w', 'ma', 'n', 'class', 'rms', 'neo']
            
            logging.info('cleaning raw data')
            dataframe= raw_dataframe[selected_columns]

            logging.info('creating features and target from cleaned data')
            features, target= dataframe.drop('neo', axis= 1).copy(), dataframe['neo'].copy()

            logging.info('data ingestion completed')

            logging.info('saving features and target data')
            pickle_file(dataframe= features, file_name= 'features.pkl')
            pickle_file(dataframe= target, file_name= 'target.pkl')
            logging.info('contents saved')
        
        except Exception as CE:
            logging.error(f'error during data ingestion: {str(CE)}', exc_info= True)
            raise CustomException(CE, sys)

if __name__ == '__main__':
    # data ingestion
    object= DataIngestion()
    cleaned_dataframe= object.data_ingestion()