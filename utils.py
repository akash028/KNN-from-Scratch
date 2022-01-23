# utils.py: Utility file for implementing helpful utility functions used by the ML algorithms.

import numpy as np
import math

def euclidean_distance(x1, x2):
    """
    Computes and returns the Euclidean distance between two vectors.

    Args:
        x1: A numpy array of shape (n_features,).
        x2: A numpy array of shape (n_features,).
    """
    return np.sqrt(np.sum(pow(np.subtract(x1, x2), 2)))
    #raise NotImplementedError('This function must be implemented by the student.')


def manhattan_distance(x1, x2):
    """
    Computes and returns the Manhattan distance between two vectors.

    Args:
        x1: A numpy array of shape (n_features,).
        x2: A numpy array of shape (n_features,).
    """
    #l = len(x1) if len(x1) > len(x2) else len(x2)

    return np.sum(abs(np.subtract(x1, x2)))

    #raise NotImplementedError('This function must be implemented by the student.')
