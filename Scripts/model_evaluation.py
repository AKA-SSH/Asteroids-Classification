import sys
from sklearn.ensemble import RandomForestClassifier
from utils.logger import logging
from utils.exception import CustomException
from utils.unpickle_file import unpickle_file
from utils.evaluate_model import evaluate_model

class EvaluateModel:
    def __init__(self) -> None:
        pass

    def model_evaluation(self, test_features_file_path, test_target_file_path, model_file_path):
        """
        Evaluate a machine learning model using the provided test features, test target, and model.

        Parameters:
        - test_features_file_path (str): File path to the pickled file containing test features.
        - test_target_file_path (str): File path to the pickled file containing test target.
        - model_file_path (str): File path to the pickled file containing the trained model.

        Returns:
        None

        Example Usage:
        ```python
        eval_model = EvaluateModel()
        eval_model.model_evaluation('path/to/test_features.pkl', 'path/to/test_target.pkl', 'path/to/model.pkl')
        ```
        """
        try:
            logging.info('Model evaluation initiated')

            test_features = unpickle_file(test_features_file_path)
            test_target = unpickle_file(test_target_file_path)
            model = unpickle_file(model_file_path)

            test_predictions = model.predict(test_features)

            evaluation_report = evaluate_model(test_target, test_predictions)
            logging.info(evaluation_report)
            logging.info('Model evaluation completed')
            return ('performance metrics in logs.')

        except Exception as CE:
            logging.error(f'Error during model evaluation: {str(CE)}', exc_info=True)
            raise CustomException(str(CE), sys)
