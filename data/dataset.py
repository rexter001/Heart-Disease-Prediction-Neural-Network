"""
Dataset loading and preprocessing.

This module:
- Loads the Heart Failure dataset
- One-hot encodes categorical features
- Normalizes numerical features
- Shuffles the dataset
- Splits into train/test sets
"""

import numpy as np

from nn.utils import (
    shuffle_data,
    train_test_split,
    normalize_features,
)


def one_hot_encode(column: np.ndarray):
    """
    One-hot encode a categorical column.

    Returns
    -------
    encoded_column : np.ndarray
    categories : np.ndarray
    """

    categories = np.unique(column)

    encoded = np.zeros((len(column), len(categories)))

    for i, category in enumerate(categories):
        encoded[:, i] = (column == category).astype(float)

    return encoded, categories


def load_dataset(file_path="data/heart.csv"):
    """
    Load and preprocess the Heart Failure dataset.

    Returns
    -------
    X_train
    X_test
    y_train
    y_test
    """

    # Load CSV (skip header)
    data = np.genfromtxt(
        file_path,
        delimiter=",",
        skip_header=1,
        dtype=str
    )

    # -----------------------------
    # Target
    # -----------------------------
    y = data[:, -1].astype(int).reshape(-1, 1)

    # -----------------------------
    # Numerical Features
    # -----------------------------
    numerical_indices = [0, 3, 4, 7, 9]

    X_numeric = data[:, numerical_indices].astype(float)

    # -----------------------------
    # Categorical Features
    # -----------------------------
    categorical_indices = [1, 2, 5, 6, 8, 10]

    encoded_columns = []

    for index in categorical_indices:
        encoded, _ = one_hot_encode(data[:, index])
        encoded_columns.append(encoded)

    X_categorical = np.hstack(encoded_columns)

    # -----------------------------
    # Combine Features
    # -----------------------------
    X = np.hstack((X_numeric, X_categorical))

    # -----------------------------
    # Normalize
    # -----------------------------
    X, mean, std = normalize_features(X)

    # -----------------------------
    # Shuffle
    # -----------------------------
    X, y = shuffle_data(X, y)

    # -----------------------------
    # Split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2
    )

    return X_train, X_test, y_train, y_test