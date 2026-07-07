"""
Loss functions used by the neural network.

This module implements Binary Cross Entropy (BCE)
for binary classification.
"""

import numpy as np


def binary_cross_entropy(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """
    Compute Binary Cross Entropy loss.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels.
        Shape: (batch_size, 1)

    y_pred : np.ndarray
        Predicted probabilities.
        Shape: (batch_size, 1)

    Returns
    -------
    float
        Mean Binary Cross Entropy loss.
    """

    # Prevent log(0)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

    loss = -np.mean(
        y_true * np.log(y_pred)
        +
        (1 - y_true) * np.log(1 - y_pred)
    )

    return loss


def binary_cross_entropy_derivative(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> np.ndarray:
    """
    Compute the derivative of Binary Cross Entropy
    with respect to the predictions.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.

    y_pred : np.ndarray
        Predicted probabilities.

    Returns
    -------
    np.ndarray
        Gradient of the loss with respect to y_pred.
    """

    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

    gradient = (
        -(y_true / y_pred)
        +
        ((1 - y_true) / (1 - y_pred))
    )

    gradient /= y_true.shape[0]

    return gradient