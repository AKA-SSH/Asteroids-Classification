import sys
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from utils.logger import logging
from utils.exception import CustomException
from utils.pickle_file import pickle_file
from utils.unpickle_file import unpickle_file

class DataPreprocessing:
    def __init__(self) -> None:
        pass

    def data_preprocessing(self, features_file_path:str, target_file_path:str):
        """
        Performs data preprocessing for machine learning tasks.

        Parameters:
        - features_file_path (str): File path to the pickled file containing features.
        - target_file_path (str): File path to the pickled file containing target labels.

        Raises:
        - CustomException: If an error occurs during the data preprocessing process.

        Returns:
        None
        """
        try:
            logging.info('Data preprocessing initiated')
            
            logging.info('Loading data')
            features= unpickle_file(features_file_path)
            target= unpickle_file(target_file_path)

            logging.info('fetching categorical features')
            categorical_columns= features.select_dtypes(include='O').columns

            logging.info('Splitting train and test data')
            train_features, test_features, train_target, test_target= train_test_split(features, target, test_size=0.2, random_state=42)

            logging.info('Performing label encoding on categorical features')
            LE= LabelEncoder()
            for column in categorical_columns:
                 train_features[column]= LE.fit_transform(train_features[column])
                 test_features[column]= LE.transform(test_features[column])

            pickle_file(object=LE, file_name='label_encoder.pkl')

            logging.info('Encoding target label')
            train_target= train_target.map({'N': 0, 'Y': 1})
            test_target= test_target.map({'N': 0, 'Y': 1})

            logging.info('Data preprocessing completed')
            logging.info('Saving train and test data')

            pickle_file(object=train_features, file_name='train_features.pkl')
            pickle_file(object=train_target, file_name='train_target.pkl')
            pickle_file(object=test_features, file_name='test_features.pkl')
            pickle_file(object=test_target, file_name='test_target.pkl')

        except Exception as CE:
            logging.error(f'Error during data preprocessing: {str(CE)}', exc_info=True)
            raise CustomException(CE, sys)