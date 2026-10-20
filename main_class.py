import numpy as np
from mathematic_functions import *

class polska_gurom(object):
    def __init__(self , sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, x:np.ndarray) -> np.ndarray:
        """
        Computes the feedforward pass of the neural network for a single input vector.

        Parameters:
        x (numpy.ndarray): The input column vector of shape (sizes[0], 1).

        Returns:
        numpy.ndarray: The output vector of the network from the final layer.

        Raises:
        ValueError: If the input 'x' does not match the expected input layer dimensions.
        """
        if x.shape != (self.sizes[0], 1):
            raise ValueError(f"Input vector 'x' must have shape ({self.sizes[0]}, 1), but got {x.shape}")

        input = x
        for i in range(len(self.weights)):
            w = self.weights[i]
            b = self.biases[i]
            z = (w @ input) + b
            input = sigmoid_function(z)
            
        return input

    def result_test_data(self, data) -> int:
        """
        Evaluates the neural network on a test dataset and returns the number of correct predictions.

        Parameters:
        data (iterable): An iterable collection of tuples (x, y) where 'x' is the input vector 
                         and 'y' is the true label/target value.

        Returns:
        int: The total count of correct predictions.

        Raises:
        ValueError: If the provided dataset is empty or None.
        RuntimeError: If an error occurs during inference or sample processing.
        """
        if data is None or len(data) == 0:
            raise ValueError("Data cannot be empty.")
            
        counter = 0
        is_binary = self.sizes[-1] == 1
        
        for x, y in data:
            try:
                output = self.feedforward(x)
                if is_binary:
                    prediction = 1 if output[0, 0] >= 0.5 else 0
                else:
                    prediction = np.argmax(output)
                    
                if prediction == y:
                    counter += 1
            except Exception as e:
                raise RuntimeError(f"Error processing test sample: {e}")
                
        return counter

    def update_mini_batch(self, mini_batch: list, eta: float) -> tuple:
        """
        Updates the network's weights and biases by applying gradient descent
        using backpropagation to a single mini-batch.

        Parameters:
        mini_batch (list): A list of tuples (x, y), where 'x' is the input column vector and 'y' is the desired output.
        eta (float): The learning rate.

        Returns:
        tuple: Updated weights and biases after applying gradient descent.
        """

        # Initialize gradients for weights and biases
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        # Perform backpropagation for each example in the mini-batch
        for x, y in mini_batch:
            delta_nabla_w, delta_nabla_b = self.backprop(x, y)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]

        # Update weights and biases using the average gradients
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]