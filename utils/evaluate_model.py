import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from tabulate import tabulate

def evaluate_model(true, pred):
    '''
    This function returns a classification report for accuracy, precision, recall, and F1 score.

    Input:
        - True Value
        - Predicted Value
    
    Output:
        - Classification Report
    '''
    try:
        true_values = true.values.ravel() if isinstance(true, pd.Series) or isinstance(true, pd.DataFrame) else true
        pred_values = pred.ravel() if isinstance(pred, np.ndarray) else pred.values.ravel()

        true_int = true_values.astype(int)
        pred_int = pred_values.astype(int)

        class_report = classification_report(true_int, pred_int, target_names=['Class 0', 'Class 1'], output_dict=True)
        class_report_table = tabulate(pd.DataFrame.from_dict(class_report), headers='keys', tablefmt='grid')

        return f"Classification Report:\n{class_report_table}"

    except Exception as e:
        return f"Error during evaluation: {str(e)}"
