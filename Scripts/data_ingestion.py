import sys
import pandas as pd

from utils.logger import logging
from utils.exception import CustomException
from utils.pickle_file import pickle_file

from Scripts.data_preprocessing import DataPreprocessing
from Scripts.feature_engineering import FeatureEngineering
from Scripts.oversampling import OverSampling

class DataIngestion:
    def __init__(self) -> None:
        pass

    def data_ingestion(self, raw_data_file_path='Data\\raw_data.csv'):
        """
        Performs the data ingestion process for space object data.

        Parameters:
        - raw_data_file_path (str): The file path to the raw data in CSV format.

        Raises:
        - CustomException: If an error occurs during the data ingestion process.

        Returns:
        - tuple: A tuple containing the cleaned features and target data.
        """
        try:
            logging.info('Data ingestion initiated')

            logging.info(f'Loading raw data from {raw_data_file_path}')
            raw_dataframe = pd.read_csv(raw_data_file_path, low_memory=False)
            selected_columns = ['epoch', 'e', 'i', 'om', 'w', 'ma', 'n', 'class', 'rms', 'neo']

            logging.info('Cleaning raw data')
            dataframe = raw_dataframe[selected_columns]
            dataframe = dataframe.dropna()

            logging.info(f'Shape after cleaning: {dataframe.shape}')

            logging.info('Creating features and target from cleaned data')
            features, target = dataframe.drop('neo', axis=1).copy(), dataframe['neo'].copy()

            logging.info(f'Shape of features: {features.shape}, Shape of target: {target.shape}')

            logging.info('Data ingestion completed')

            logging.info('Saving features and target data')
            pickle_file(object=features, file_name='features.pkl')
            pickle_file(object=target, file_name='target.pkl')
            logging.info('Contents saved')

        except Exception as CE:
            logging.error(f'Error during data ingestion: {str(CE)}', exc_info=True)
            raise CustomException(CE, sys)

if __name__ == '__main__':
    # Data ingestion
    data_ingestion_object = DataIngestion()
    data_ingestion_object.data_ingestion()

    # Data preprocessing
    data_preprocessing_object = DataPreprocessing()
    data_preprocessing_object.data_preprocessing(features_file_path='artifacts\\features.pkl', target_file_path='artifacts\\target.pkl')

    # Feature Engineering
    feature_engineering_object = FeatureEngineering()
    feature_engineering_object.engineer_feature(train_features_file_path='artifacts\\train_features.pkl', test_features_file_path='artifacts\\test_features.pkl', train_target_file_path='artifacts\\train_target.pkl', test_target_file_path='artifacts\\test_target.pkl')

    # Over Sampling
    over_sampling_object = OverSampling()
    over_sampling_object.over_sampler(train_features_file_path='artifacts\\train_features.pkl', train_target_file_path='artifacts\\test_features.pkl')

    # Model Training
