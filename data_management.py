import random

def batch_split(training_data: list, mini_batch_size: int = 10) -> list:
    """
    Shuffles the training data and splits it into mini-batches.

    Parameters:
    training_data (list): A list of tuples (x, y), where 'x' is the input column vector and 'y' is the desired output.
    mini_batch_size (int): Number of examples per mini-batch (the last one may be smaller).

    Returns:
    list: A list of mini-batches, each a list of (x, y) tuples.
    """

    if mini_batch_size <= 0:
        raise ValueError("mini_batch_size must be positive")

    data = list(training_data)
    random.shuffle(data)

    return [data[k:k + mini_batch_size] for k in range(0, len(data), mini_batch_size)]