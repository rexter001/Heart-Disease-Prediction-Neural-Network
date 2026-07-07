"""
Utility functions used throughout the project.

This module contains helper functions for:

- Dataset preprocessing
- Dataset splitting
- Dataset shuffling
- Mini-batch creation
- Feature normalization
"""

import numpy as np


def shuffle_data(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42
):
    """
    Shuffle features and labels together.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.

    y : np.ndarray
        Target vector.

    random_state : int
        Seed for reproducibility.

    Returns
    -------
    tuple
        Shuffled X and y.
    """

    np.random.seed(random_state)

    indices = np.random.permutation(len(X))

    return X[indices], y[indices]


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2
):
    """
    Split dataset into training and testing sets.

    Parameters
    ----------
    X : np.ndarray
        Features.

    y : np.ndarray
        Labels.

    test_size : float
        Fraction of testing samples.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """

    split_index = int(len(X) * (1 - test_size))

    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]

    return X_train, X_test, y_train, y_test


def normalize_features(X: np.ndarray):
    """
    Normalize features using z-score normalization.

    Formula

    X = (X - mean) / std

    Returns
    -------
    normalized_data
    mean
    std
    """

    mean = np.mean(X, axis=0)

    std = np.std(X, axis=0)

    # Avoid division by zero
    std[std == 0] = 1

    X_normalized = (X - mean) / std

    return X_normalized, mean, std


def create_mini_batches(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int
):
    """
    Create mini-batches.

    Returns
    -------
    list of tuples
    """

    mini_batches = []

    for i in range(0, len(X), batch_size):

        X_batch = X[i:i + batch_size]

        y_batch = y[i:i + batch_size]

        mini_batches.append((X_batch, y_batch))

    return mini_batches


def accuracy_score(
    y_true: np.ndarray,
    y_pred: np.ndarray
):
    """
    Compute classification accuracy.
    """

    return np.mean(y_true == y_pred)

def precision_score(y_true, y_pred):
    """
    Compute precision.

    Precision = TP / (TP + FP)
    """

    tp = np.sum((y_true == 1) & (y_pred == 1))

    fp = np.sum((y_true == 0) & (y_pred == 1))

    if tp + fp == 0:
        return 0.0

    return tp / (tp + fp)


def recall_score(y_true, y_pred):
    """
    Compute recall.

    Recall = TP / (TP + FN)
    """

    tp = np.sum((y_true == 1) & (y_pred == 1))

    fn = np.sum((y_true == 1) & (y_pred == 0))

    if tp + fn == 0:
        return 0.0

    return tp / (tp + fn)


def f1_score(y_true, y_pred):
    """
    Compute F1 Score.
    """

    precision = precision_score(y_true, y_pred)

    recall = recall_score(y_true, y_pred)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def confusion_matrix(y_true, y_pred):
    """
    Compute confusion matrix.

    Returns
    -------
    [[TN, FP],
     [FN, TP]]
    """

    tp = np.sum((y_true == 1) & (y_pred == 1))

    tn = np.sum((y_true == 0) & (y_pred == 0))

    fp = np.sum((y_true == 0) & (y_pred == 1))

    fn = np.sum((y_true == 1) & (y_pred == 0))

    return np.array([
        [tn, fp],
        [fn, tp]
    ])