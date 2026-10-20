import numpy as np

def sigmoid_function(x:float | np.ndarray) -> float | np.ndarray:
    """
    Computes the sigmoid activation function.
    
    Parameters:
    x (float or numpy.ndarray): Input value, vector, or matrix.
    
    Returns:
    float or numpy.ndarray: Output values mapped to the range (0, 1).
    """
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x:float | np.ndarray) -> float | np.ndarray:
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

def cost_function_derivative(output_activations: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Computes the derivative of the quadratic cost function

    Parameters:
    output_activations (np.ndarray): Output of the output layer. Range [0,1]
    y (np.ndarray): Correct output of the output layer. What it should look like. Range [0,1]
    
    Returns:
    Derivatives (np.ndarray)    
    """
    return (output_activations - y)