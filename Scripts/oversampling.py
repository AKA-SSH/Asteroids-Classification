import sys
from imblearn.over_sampling import RandomOverSampler
from utils.logger import logging
from utils.exception import CustomException
from utils.pickle_file import pickle_file
from utils.unpickle_file import unpickle_file

class OverSampling:
    def __init__(self) -> None:
        pass

    def over_sampler(self, train_features_file_path, train_target_file_path):
        try:
            logging.info('Random over-sampling initiated')

            logging.info('Loading training features and target')
            train_features = unpickle_file(train_features_file_path)
            train_target = unpickle_file(train_target_file_path)

            train_target = train_target.values.ravel()
            train_target = train_target.astype(int)

            logging.info(f'Shape of train_features: {train_features.shape}')
            logging.info(f'Shape of train_target: {train_target.shape}')

            logging.info('Converting train_target to Pandas Series')
            train_target = pd.Series(train_target)

            logging.info('Applying random over-sampling')
            ROS = RandomOverSampler(sampling_strategy='minority', random_state=42)
            resampled_train_features, resampled_train_target = ROS.fit_resample(train_features, train_target.sample(n= train_features.shape[0], random_state= 42))

            logging.info('Saving resampled data')
            pickle_file(object=resampled_train_features, file_name='resampled_train_features.pkl')
            pickle_file(object=resampled_train_target, file_name='resampled_train_target.pkl')

            logging.info('Random over-sampling completed')

        except Exception as CE:
            logging.error(f'Error during random over-sampling: {str(CE)}', exc_info=True)
            raise CustomException(str(CE))

# Assuming you have missed importing pandas at the beginning of your script
import pandas as pd
