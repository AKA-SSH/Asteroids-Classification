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
        """
        Applies Random Over-sampling to balance the class distribution in the training data.

        Parameters:
        - train_features_file_path (str): File path to the pickled file containing training features.
        - train_target_file_path (str): File path to the pickled file containing training target labels.

        Raises:
        - CustomException: If an error occurs during the over-sampling process.

        Returns:
        None
        """
        try:
            logging.info('random over-sampling initiated')

            logging.info('loading training features and target')
            train_features= unpickle_file(train_features_file_path)
            train_target= unpickle_file(train_target_file_path)

            logging.info('applying random over-sampling')
            ROS= RandomOverSampler(sampling_strategy='minority', random_state=42)
            resampled_train_features, resampled_train_target= ROS.fit_resample(train_features, train_target)

            logging.info('saving random over-sampler and resampled data')
            pickle_file(object=ROS, file_name='random_over_sampler.pkl')
            pickle_file(object=resampled_train_features, file_name='resampled_train_features.pkl')
            pickle_file(object=resampled_train_target, file_name='resampled_train_target.pkl')

            logging.info('random over-sampling completed')

        except Exception as CE:
            logging.error(f'error during random over-sampling: {str(CE)}', exc_info=True)
            raise CustomException(CE, sys)