from random import random

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
    
    def backprop(self, x: np.ndarray, y: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Parameters:
        x (np.ndarray) - An array of the activations for the input of the network
        y (np.ndarray) - An array with outputs for x
        
        Returns: 
        tuple: A tuple containing two lists of np.ndarray, the gradients.
        
        Raises:
        ValueError: If the input 'x' or 'y' does not match the expected input layer dimensions.
        """
        
        if x.shape != (self.sizes[0], 1):
            raise ValueError(f"Input 'x' must have shape ({self.sizes[0]}, 1), got {x.shape}")
        if y.shape != (self.sizes[-1], 1):
            raise ValueError(f"Target 'y' must have shape ({self.sizes[-1]}, 1), got {y.shape}")
        
        grad_w = [np.zeros(w.shape) for w in self.weights]
        grad_b = [np.zeros(b.shape) for b in self.biases]
        
        input = x 
        activations = [x]
        zs = []
        
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, input) + b
            zs.append(z)
            input = sigmoid_function(z)
            activations.append(input)
        
        error = cost_function_derivative(activations[-1], y) * sigmoid_derivative(sigmoid_function(zs[-1]))
        grad_b[-1] = error
        grad_w[-1] = np.dot(error, activations[-2].T)
        
        for l in range(2, self.num_layers):
            error = np.dot(self.weights[-l+1].T, error) * sigmoid_derivative(sigmoid_function(zs[-l]))
            grad_b[-l] = error
            grad_w[-l] = np.dot(error, activations[-l-1].T)
        
        return (grad_b, grad_w)

    def train(self, training_data: list[tuple[np.ndarray,np.ndarray]], epochs: int, batch_size: int, learning_rate: float, test_data: list[tuple[np.ndarray, int]] = None):
        """

        Parameters:
            training_data (list[tuple[np.ndarray,np.ndarray]]): A list of tuples (x, y), where 'x' is the input column vector and 'y' is the desired output.
            epochs (int): Number of epochs for how long the training must go.
            batch_size (int): Size of the mini batches.
            learning_rate (float): Learning rate
            test_data (list[tuple[np.ndarray, int]], optional): A list of tuples (x, y), where 'x' is the input column vector and 'y' is the desired output.
            It is used to compare the training results with the correct ones. Using it slows down the algorithm. Defaults to None.
        
        Raises:
            ValueError: If the input 'training_data' is empty.
            ValueError: If the number of epochs is less or equal than 0.
            ValueError: If the value of the learning rate is less than zero.
        """
        
        if not training_data:
            raise ValueError("Training Data shouldn't be empty")
        if epochs <= 0:
            raise ValueError(f"Epochs must be an integer greater than 0, got {epochs}")
        if learning_rate <= 0:
            raise ValueError(f"Learning rate must be greater than 0, got {learning_rate}")
        
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        
        
        for i in range(epochs):
            
            batches = self.batch_split(training_data, batch_size)    
            for batch in batches:
                self.update_mini_batch(batch, learning_rate)
            
            if test_data:
                print(f"Epoch {i}: {self.result_test_data(test_data)} / {n_test}")
            else:
                print(f"Epoch {i} completed")