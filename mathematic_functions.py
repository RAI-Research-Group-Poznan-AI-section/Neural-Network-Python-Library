import numpy as np

def sigmoid_function(x):
    """
    Computes the sigmoid activation function.
    
    Parameters:
    x (float or numpy.ndarray): Input value, vector, or matrix.
    
    Returns:
    float or numpy.ndarray: Output values mapped to the range (0, 1).
    """
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    """
    Computes the derivative of the sigmoid function.
    
    Note: This function assumes that 'x' is already the output of the 
    sigmoid function (i.e., s = sigmoid(x)), which is the standard and 
    most efficient approach for backpropagation.
    
    Parameters:
    x (float or numpy.ndarray): Output of the sigmoid function, in the range [0, 1].
    
    Returns:
    float or numpy.ndarray: Derivative values.
    
    Raises:
    ValueError: If any value of 'x' is outside the valid range [0, 1].
    """
    
    if np.any(x < 0) or np.any(x > 1):
        raise ValueError("Input 'x' must be in the range [0, 1]. Did you pass the raw input instead of the sigmoid output?")
        
    return x * (1 - x)