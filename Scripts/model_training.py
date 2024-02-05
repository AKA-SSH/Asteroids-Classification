import sys
from sklearn.ensemble import RandomForestClassifier
from utils.logger import logging
from utils.exception import CustomException
from utils.pickle_file import pickle_file
from utils.unpickle_file import unpickle_file

class ModelTrainer:
    def __init__(self) -> None:
        pass

    def model_trainer(self, resampled_train_features_file_path, resampled_train_target_file_path):
        """
        Train a RandomForestClassifier model.

        Parameters:
        - resampled_train_features_file_path (str): File path to the pickled file containing resampled training features.
        - resampled_train_target_file_path (str): File path to the pickled file containing resampled training target.

        Raises:
        - CustomException: If an error occurs during the model training process.

        Returns:
        None

        Example Usage:
        ```python
        trainer = ModelTrainer()
        trainer.model_trainer('path/to/resampled_train_features.pkl', 'path/to/resampled_train_target.pkl')
        ```
        """
        try:
            logging.info('Model training initiated')

            logging.info('Loading resampled training features and target')
            train_features = unpickle_file(resampled_train_features_file_path)
            train_target = unpickle_file(resampled_train_target_file_path)

            logging.info('Training RandomForestClassifier')
            RFC = RandomForestClassifier()
            RFC.fit(train_features, train_target)

            logging.info('Saving trained model')
            pickle_file(object=RFC, file_name='model.pkl')

            logging.info('Model training completed')

        except Exception as CE:
            logging.error(f'Error during model training: {str(CE)}', exc_info=True)
            raise CustomException(CE, sys)
