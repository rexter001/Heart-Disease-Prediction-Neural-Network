"""
Activation functions used by the neural network.

This module contains:
- ReLU (hidden layers)
- Sigmoid (output layer)
- Their corresponding derivatives for backpropagation
"""

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    """
    Apply the ReLU activation function.

    Parameters
    ----------
    x : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Output after applying ReLU.
    """
    return np.maximum(0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    """
    Compute the derivative of ReLU.

    Parameters
    ----------
    x : np.ndarray
        Input array before activation.

    Returns
    -------
    np.ndarray
        Derivative of ReLU.
    """
    return (x > 0).astype(float)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Apply the sigmoid activation function.

    Parameters
    ----------
    x : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Values between 0 and 1.
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(sigmoid_output: np.ndarray) -> np.ndarray:
    """
    Compute the derivative of the sigmoid function.

    Parameters
    ----------
    sigmoid_output : np.ndarray
        Output obtained after applying sigmoid.

    Returns
    -------
    np.ndarray
        Derivative of sigmoid.
    """
    return sigmoid_output * (1 - sigmoid_output)