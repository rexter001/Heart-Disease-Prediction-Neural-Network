"""
Neural Network implementation using only NumPy.

Architecture:

Input
    ↓
Dense
    ↓
ReLU
    ↓
Dense
    ↓
ReLU
    ↓
Dense
    ↓
Sigmoid
"""

import numpy as np

from nn.layers import DenseLayer
from nn.activations import (
    relu,
    relu_derivative,
    sigmoid,
)

from nn.losses import binary_cross_entropy


class NeuralNetwork:
    """
    Three-layer feedforward neural network.

    Architecture:

    Input
        ↓
    Hidden Layer 1
        ↓
    Hidden Layer 2
        ↓
    Output Layer
    """

    def __init__(
        self,
        input_size: int,
        hidden_size1: int = 16,
        hidden_size2: int = 8,
        learning_rate: float = 0.01,
    ):
        """
        Initialize the neural network.

        Parameters
        ----------
        input_size : int
            Number of input features.

        hidden_size1 : int
            Number of neurons in hidden layer 1.

        hidden_size2 : int
            Number of neurons in hidden layer 2.

        learning_rate : float
            Gradient descent learning rate.
        """

        self.learning_rate = learning_rate

        # Hidden Layer 1
        self.layer1 = DenseLayer(
            input_size=input_size,
            output_size=hidden_size1,
            initialization="he"
        )

        # Hidden Layer 2
        self.layer2 = DenseLayer(
            input_size=hidden_size1,
            output_size=hidden_size2,
            initialization="he"
        )

        # Output Layer
        self.output_layer = DenseLayer(
            input_size=hidden_size2,
            output_size=1,
            initialization="xavier"
        )

        # Training history
        self.loss_history = []

        # Cached activations
        self.z1 = None
        self.a1 = None

        self.z2 = None
        self.a2 = None

        self.z3 = None
        self.a3 = None
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Perform forward propagation through the network.

        Parameters
        ----------
        X : np.ndarray
            Input feature matrix.

        Returns
        -------
        np.ndarray
            Predicted probabilities.
        """

        # ---------- Hidden Layer 1 ----------
        self.z1 = self.layer1.forward(X)
        self.a1 = relu(self.z1)

        # ---------- Hidden Layer 2 ----------
        self.z2 = self.layer2.forward(self.a1)
        self.a2 = relu(self.z2)

        # ---------- Output Layer ----------
        self.z3 = self.output_layer.forward(self.a2)
        self.a3 = sigmoid(self.z3)

        return self.a3
    
    def backward(self, X: np.ndarray, y: np.ndarray):
        """
        Perform backpropagation and update network parameters.

        Parameters
        ----------
        X : np.ndarray
            Input feature matrix.

        y : np.ndarray
            True labels.
        """

        # Number of samples in the current batch
        m = X.shape[0]

        # =====================================================
        # Output Layer
        # =====================================================

        # Gradient of loss w.r.t. output layer pre-activation
        dZ3 = self.a3 - y

        # Backpropagate through the output layer
        dA2 = self.output_layer.backward(dZ3)

        # =====================================================
        # Hidden Layer 2
        # =====================================================

        # Apply ReLU derivative
        dZ2 = dA2 * relu_derivative(self.z2)

        # Backpropagate through layer 2
        dA1 = self.layer2.backward(dZ2)

        # =====================================================
        # Hidden Layer 1
        # =====================================================

        dZ1 = dA1 * relu_derivative(self.z1)

        self.layer1.backward(dZ1)

        # =====================================================
        # Gradient Descent
        # =====================================================

        self.output_layer.update_parameters(self.learning_rate)
        self.layer2.update_parameters(self.learning_rate)
        self.layer1.update_parameters(self.learning_rate)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: bool = True,
    ):
        """
        Train the neural network using mini-batch gradient descent.

        Parameters
        ----------
        X_train : np.ndarray
            Training feature matrix.

        y_train : np.ndarray
            Training labels.

        epochs : int
            Number of training iterations.

        batch_size : int
            Number of samples per mini-batch.

        verbose : bool
            Whether to print training progress.
        """

        from nn.utils import shuffle_data, create_mini_batches

        self.loss_history = []

        for epoch in range(epochs):

            # ----------------------------------------
            # Shuffle the dataset before each epoch
            # ----------------------------------------
            X_shuffled, y_shuffled = shuffle_data(
                X_train,
                y_train,
                random_state=epoch
            )

            # ----------------------------------------
            # Create mini-batches
            # ----------------------------------------
            mini_batches = create_mini_batches(
                X_shuffled,
                y_shuffled,
                batch_size
            )

            epoch_loss = 0.0

            # ----------------------------------------
            # Train on each mini-batch
            # ----------------------------------------
            for X_batch, y_batch in mini_batches:

                # Forward propagation
                predictions = self.forward(X_batch)

                # Compute batch loss
                batch_loss = binary_cross_entropy(
                    y_batch,
                    predictions
                )

                epoch_loss += batch_loss

                # Backpropagation
                self.backward(X_batch, y_batch)

            # ----------------------------------------
            # Average loss for the epoch
            # ----------------------------------------
            epoch_loss /= len(mini_batches)

            self.loss_history.append(epoch_loss)

            # ----------------------------------------
            # Display progress
            # ----------------------------------------
            if verbose:
                print(
                    f"Epoch [{epoch + 1}/{epochs}] "
                    f"Loss: {epoch_loss:.6f}"
                )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Parameters
        ----------
        X : np.ndarray
            Input feature matrix.

        Returns
        -------
        np.ndarray
            Probability of Heart Disease.
        """

        return self.forward(X)

    def predict(
        self,
        X: np.ndarray,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Predict binary class labels.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.

        threshold : float
            Classification threshold.

        Returns
        -------
        np.ndarray
            Predicted class labels.
        """

        probabilities = self.predict_proba(X)

        return (probabilities >= threshold).astype(int)
    
    def save(self, filepath: str):
        """
        Save neural network weights and biases.

        Parameters
        ----------
        filepath : str
            Path where model weights will be saved.
        """

        np.savez(
            filepath,

            # Layer 1
            layer1_weights=self.layer1.weights,
            layer1_bias=self.layer1.bias,

            # Layer 2
            layer2_weights=self.layer2.weights,
            layer2_bias=self.layer2.bias,

            # Output Layer
            output_weights=self.output_layer.weights,
            output_bias=self.output_layer.bias
        )


    def load(self, filepath: str):
        """
        Load neural network weights and biases.

        Parameters
        ----------
        filepath : str
            Path of saved model weights.
        """

        data = np.load(filepath)

        # Layer 1
        self.layer1.weights = data["layer1_weights"]
        self.layer1.bias = data["layer1_bias"]

        # Layer 2
        self.layer2.weights = data["layer2_weights"]
        self.layer2.bias = data["layer2_bias"]

        # Output Layer
        self.output_layer.weights = data["output_weights"]
        self.output_layer.bias = data["output_bias"]