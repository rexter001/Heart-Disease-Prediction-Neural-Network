"""
Dense (fully connected) layer implementation.

Responsibilities:
- Weight initialization
- Forward propagation
- Backward propagation
- Gradient descent parameter updates
"""

import numpy as np


class DenseLayer:
    """
    Fully connected neural network layer.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        initialization: str = "he"
    ):

        if initialization == "he":
            self.weights = (
                np.random.randn(input_size, output_size)
                * np.sqrt(2 / input_size)
            )

        elif initialization == "xavier":
            self.weights = (
                np.random.randn(input_size, output_size)
                * np.sqrt(1 / input_size)
            )

        else:
            raise ValueError(
                "Initialization must be 'he' or 'xavier'."
            )

        self.bias = np.zeros((1, output_size))

        # Cached values
        self.input = None
        self.output = None

        # Gradients
        self.d_weights = None
        self.d_bias = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward propagation.

        Z = XW + b
        """

        self.input = inputs
        self.output = np.dot(inputs, self.weights) + self.bias

        return self.output

    def backward(self, d_output: np.ndarray) -> np.ndarray:
        """
        Backward propagation.

        Parameters
        ----------
        d_output
            Gradient coming from the next layer.

        Returns
        -------
        np.ndarray
            Gradient to pass to the previous layer.
        """

        batch_size = self.input.shape[0]

        # dL/dW
        self.d_weights = np.dot(self.input.T, d_output) / batch_size

        # dL/db
        self.d_bias = np.sum(d_output, axis=0, keepdims=True) / batch_size

        # dL/dX
        d_input = np.dot(d_output, self.weights.T)

        return d_input

    def update_parameters(self, learning_rate: float):
        """
        Gradient Descent update.
        """

        self.weights -= learning_rate * self.d_weights
        self.bias -= learning_rate * self.d_bias