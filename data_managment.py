import numpy as np

def batch_split(data:np.ndarray, batch_size:int) -> np.ndarray:
    """
    Splits the data into batches of specified size.

    Parameters:
    data (np.ndarray): The input data to be split into batches.
    batch_size (int): The size of each batch.

    Returns:
    np.ndarray: An array of batches, where each batch is a list of data items.
    """
    batches = np.ndarray([data[i:i + batch_size] for i in range(0, len(data), batch_size)])
    np.shuffle(batches)
    
    return batches