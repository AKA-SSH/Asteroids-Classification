import os
import pickle

def pickle_file(object, file_name, folder_name='artifacts'):
    """
    Pickle a Pandas DataFrame and save it to a specified folder.

    Parameters:
    - object: Content to be stored.
    - file_name (str): The name of the pickled file.
    - folder_name (str, optional): The folder where the pickled file will be stored. Defaults to 'artifacts'.

    Returns:
    None

    Example:
    ```python
    pickle_file(my_dataframe, 'my_pickled_file.pkl', folder_name='MyData')
    ```

    This function creates the specified folder (if it doesn't exist), constructs the full path to the pickled file,
    and saves the DataFrame using the pickle module.

    Note: Pickling is a way to serialize and save Python objects, including Pandas DataFrames, to disk.
    """
    os.makedirs(folder_name, exist_ok=True)
    full_path = os.path.join(folder_name, file_name)
    with open(full_path, 'wb') as f:
        pickle.dump(object, f)
        print(f"{object} pickled and saved to: {full_path}")
