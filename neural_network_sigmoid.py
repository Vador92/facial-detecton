import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import os
import time
import numpy as np

# from sklearn.metrics import accuracy_score
# This file is for the sigmoid function and triple layer neural network

class Sigmoid:
    def __init__(self):
        pass

class SigmoidNeuralNetwork:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, learning_rate=0.01):
        """
        Initialize a 3-layer neural network with sigmoid activation functions
        
        Args:
            input_size: Number of input features
            hidden1_size: Number of neurons in first hidden layer
            hidden2_size: Number of neurons in second hidden layer
            output_size: Number of output classes
            learning_rate: Learning rate for gradient descent
        """
        # Initialize weights with Xavier/Glorot initialization
        self.weights1 = np.random.randn(input_size, hidden1_size) * np.sqrt(2 / (input_size + hidden1_size))
        self.bias1 = np.zeros((1, hidden1_size))
        
        self.weights2 = np.random.randn(hidden1_size, hidden2_size) * np.sqrt(2 / (hidden1_size + hidden2_size))
        self.bias2 = np.zeros((1, hidden2_size))
        
        self.weights3 = np.random.randn(hidden2_size, output_size) * np.sqrt(2 / (hidden2_size + output_size))
        self.bias3 = np.zeros((1, output_size))
        
        self.learning_rate = learning_rate
        
        # For momentum
        self.velocity_w1 = np.zeros_like(self.weights1)
        self.velocity_b1 = np.zeros_like(self.bias1)
        self.velocity_w2 = np.zeros_like(self.weights2)
        self.velocity_b2 = np.zeros_like(self.bias2)
        self.velocity_w3 = np.zeros_like(self.weights3)
        self.velocity_b3 = np.zeros_like(self.bias3)
        self.momentum = 0.9
    
    def sigmoid(self, x):
        """Sigmoid activation function with numerical stability"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function"""
        return x * (1 - x)
    
    def forward(self, X):
        """
        Forward pass through the network
        
        Args:
            X: Input data (batch_size, input_size)
            
        Returns:
            Output activations
        """
        # First hidden layer with sigmoid
        self.z1 = np.dot(X, self.weights1) + self.bias1
        self.a1 = self.sigmoid(self.z1)
        
        # Second hidden layer with sigmoid
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.a2 = self.sigmoid(self.z2)
        
        # Output layer with sigmoid
        self.z3 = np.dot(self.a2, self.weights3) + self.bias3
        self.a3 = self.sigmoid(self.z3)
        
        return self.a3
    
    def backward(self, X, y, output):
        """
        Backward pass with momentum
        
        Args:
            X: Input data (batch_size, input_size)
            y: True labels (batch_size,)
            output: Predicted probabilities from forward pass (batch_size, output_size)
        """
        batch_size = X.shape[0]
        
        # Convert y to one-hot encoding
        y_one_hot = np.zeros((batch_size, output.shape[1]))
        y_one_hot[np.arange(batch_size), y] = 1
        
        # Output layer error
        delta3 = (output - y_one_hot) * self.sigmoid_derivative(output)
        
        # Second hidden layer error
        delta2 = np.dot(delta3, self.weights3.T) * self.sigmoid_derivative(self.a2)
        
        # First hidden layer error
        delta1 = np.dot(delta2, self.weights2.T) * self.sigmoid_derivative(self.a1)
        
        # Calculate gradients
        dW3 = np.dot(self.a2.T, delta3) / batch_size
        db3 = np.sum(delta3, axis=0, keepdims=True) / batch_size
        dW2 = np.dot(self.a1.T, delta2) / batch_size
        db2 = np.sum(delta2, axis=0, keepdims=True) / batch_size
        dW1 = np.dot(X.T, delta1) / batch_size
        db1 = np.sum(delta1, axis=0, keepdims=True) / batch_size
        
        # Update with momentum
        self.velocity_w3 = self.momentum * self.velocity_w3 - self.learning_rate * dW3
        self.velocity_b3 = self.momentum * self.velocity_b3 - self.learning_rate * db3
        self.velocity_w2 = self.momentum * self.velocity_w2 - self.learning_rate * dW2
        self.velocity_b2 = self.momentum * self.velocity_b2 - self.learning_rate * db2
        self.velocity_w1 = self.momentum * self.velocity_w1 - self.learning_rate * dW1
        self.velocity_b1 = self.momentum * self.velocity_b1 - self.learning_rate * db1
        
        self.weights3 += self.velocity_w3
        self.bias3 += self.velocity_b3
        self.weights2 += self.velocity_w2
        self.bias2 += self.velocity_b2
        self.weights1 += self.velocity_w1
        self.bias1 += self.velocity_b1
    
    def train(self, X, y, epochs=100, batch_size=32, verbose=True):
        """
        Train the neural network
        
        Args:
            X: Training data (num_samples, input_size)
            y: Training labels (num_samples,)
            epochs: Number of training epochs
            batch_size: Size of mini-batches
            verbose: Whether to print progress
        """
        num_samples = X.shape[0]
        losses = []
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(num_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Mini-batch gradient descent
            for i in range(0, num_samples, batch_size):
                end = min(i + batch_size, num_samples)
                X_batch = X_shuffled[i:end]
                y_batch = y_shuffled[i:end]
                
                # Forward pass
                output = self.forward(X_batch)
                
                # Backward pass
                self.backward(X_batch, y_batch, output)
            
            # Compute loss for monitoring (optional)
            if verbose and epoch % 5 == 0:
                output = self.forward(X)
                y_one_hot = np.zeros((num_samples, output.shape[1]))
                y_one_hot[np.arange(num_samples), y] = 1
                loss = -np.sum(y_one_hot * np.log(np.clip(output, 1e-15, 1.0))) / num_samples
                losses.append(loss)
                
                # Calculate accuracy on training data
                predictions = np.argmax(output, axis=1)
                accuracy = np.mean(predictions == y)
                
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Training Accuracy: {accuracy:.4f}")
        
        return losses
    
    def predict(self, X):
        """
        Make predictions for given inputs
        
        Args:
            X: Input data (num_samples, input_size)
            
        Returns:
            Predicted class labels
        """
        output = self.forward(X)
        return np.argmax(output, axis=1)
    
    def evaluate(self, X, y):
        """
        Evaluate accuracy on given data
        
        Args:
            X: Input data
            y: True labels
            
        Returns:
            Accuracy score
        """
        predictions = self.predict(X)
        return metrics.accuracy_score(y, predictions)

    
