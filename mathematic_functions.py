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

class QuadraticCost:
    
    @staticmethod
    def fn(output_activations: np.ndarray, y: np.ndarray) -> float:
        """ 
        Computes the raw cost value, not used in the calculations in the network, may be used to monitor the training process.
        
        Parameters:
        output_activations (np.ndarray): Output of the output layer. Range [0,1]
        y (np.ndarray): Correct output of the output layer. What it should look like. Range [0,1]
        
        Returns:
        float: error value
        """
        return 0.5 * np.linalg.norm(output_activations - y)**2
    
    @staticmethod
    def delta(z: np.ndarray, output_activations: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the quadratic cost function
    
        Parameters:
        z (np.ndarray): Weighted inputs of the output layer.
        output_activations (np.ndarray): Output of the output layer. Range [0,1]
        y (np.ndarray): Correct output of the output layer. What it should look like. Range [0,1]
        
        Returns:
        Derivatives (np.ndarray)    
        """
        return (output_activations - y) * sigmoid_derivative(sigmoid_function(z))
    
class CrossEntropyCost:
    
    @staticmethod
    def fn(output_activations: np.ndarray, y: np.ndarray) -> float:
        """ 
        Computes the raw cost value, not used in the calculations in the network, may be used to monitor the training process.
        
        Parameters:
        output_activations (np.ndarray): Output of the output layer. Range [0,1]
        y (np.ndarray): Correct output of the output layer. What it should look like. Range [0,1]
        
        Returns:
        float: error value
        """
        return np.sum(np.nan_to_num(-y * np.log(output_activations) - (1 - y) * np.log(1 - output_activations)))
    
    @staticmethod
    def delta(z: np.ndarray, output_activations: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the cross entropy cost function
    
        Parameters:
        z (np.ndarray): Weighted inputs of the output layer. It is not used here, just for the sake of generalization.
        output_activations (np.ndarray): Output of the output layer. Range [0,1]
        y (np.ndarray): Correct output of the output layer. What it should look like. Range [0,1]
        
        Returns:
        Derivatives (np.ndarray)    
        """
        return (output_activations - y)
    