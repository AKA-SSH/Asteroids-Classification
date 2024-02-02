import pickle

def unpickle_file(file_name):
    """
    Unpickle a previously pickled file and return the Python object.

    Parameters:
    - file_name (str): The name of the pickled file to be unpickled.

    Returns:
    - data: The Python object loaded from the pickled file.

    Example:
    ```python
    my_data = unpickle_file('my_pickled_file.pkl')
    ```

    This function reads a pickled file using the pickle module and returns the corresponding Python object.
    """
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data