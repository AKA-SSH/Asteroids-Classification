import sys
from sklearn.cluster import MiniBatchKMeans

from utils.logger import logging
from utils.exception import CustomException
from utils.pickle_file import pickle_file
from utils.unpickle_file import unpickle_file

class FeatureEngineering:
    def __init__(self) -> None:
        pass

    def engineer_feature(self, train_features_file_path: str, test_features_file_path: str):
        """
        Performs feature engineering using MiniBatchKMeans clustering.

        Parameters:
        - train_features_file_path (str): File path to the pickled file containing training features.
        - test_features_file_path (str): File path to the pickled file containing test features.

        Raises:
        - CustomException: If an error occurs during the feature engineering process.

        Returns:
        None
        """
        try:
            logging.info('feature engineering initiated')

            logging.info('loading training and test features')
            train_features= unpickle_file(train_features_file_path)
            test_features= unpickle_file(test_features_file_path)

            logging.info('applying MiniBatchKMeans clustering')
            optimal_clusters= 6
            kmeans_clustering= MiniBatchKMeans(n_clusters=optimal_clusters, random_state=42)
            train_cluster_labels= kmeans_clustering.fit_predict(train_features)
            test_cluster_labels= kmeans_clustering.predict(test_features)

            pickle_file(object=kmeans_clustering, file_name='kmeans_clustering.pkl')

            logging.info('adding cluster labels to features')
            train_features['cluster']= train_cluster_labels
            test_features['cluster']= test_cluster_labels

            logging.info('saving engineered features')
            pickle_file(object=train_features, file_name='train_features.pkl')
            pickle_file(object=test_features, file_name='test_features.pkl')

            logging.info('feature engineering completed')

        except Exception as CE:
            logging.error(f'Error during feature engineering: {str(CE)}', exc_info=True)
            raise CustomException(CE, sys)