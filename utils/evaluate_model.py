from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tabulate import tabulate

def evaluate_model(true, pred):
    '''
    This function returns an evaluation table with accuracy, precision, recall, and F1 score.

    Input:
        - True Value
        - Predicted Value
    
    Output:
        - Evaluation Table
    '''
    ACC = accuracy_score(true, pred)
    PRE = precision_score(true, pred, average='weighted')  # 'weighted' for unbalanced target classes
    REC = recall_score(true, pred, average='weighted')
    F1 = f1_score(true, pred, average='weighted')
    
    table = [['ACC', ACC],
             ['PRE', PRE],
             ['REC', REC],
             ['F1', F1]]
    
    evaluation = tabulate(table, headers=['METRIC', 'SCORE'], tablefmt='grid')
    
    # Adding classification report for more detailed metrics
    class_report = classification_report(true, pred, target_names=['Class 0', 'Class 1'], output_dict=True)
    class_report_table = tabulate(class_report, headers='keys', tablefmt='grid')

    return f"{evaluation}\n\nClassification Report:\n{class_report_table}"
