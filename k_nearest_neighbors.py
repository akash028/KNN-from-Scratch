# k_nearest_neighbors.py: Machine learning implementation of a K-Nearest Neighbors classifier from scratch.
#
# Submitted by: Akash Bhapkar abhapkar@iu.edu
#
# Based on skeleton code by CSCI-B 551 Fall 2021 Course Staff

import numpy as np
from utils import euclidean_distance, manhattan_distance
from collections import Counter

class KNearestNeighbors:
    """
    A class representing the machine learning implementation of a K-Nearest Neighbors classifier from scratch.

    Attributes:
        n_neighbors
            An integer representing the number of neighbors a sample is compared with when predicting target class
            values.

        weights
            A string representing the weight function used when predicting target class values. The possible options are
            {'uniform', 'distance'}.

        _X
            A numpy array of shape (n_samples, n_features) representing the input data used when fitting the model and
            predicting target class values.

        _y
            A numpy array of shape (n_samples,) representing the true class values for each sample in the input data
            used when fitting the model and predicting target class values.

        _distance
            An attribute representing which distance metric is used to calculate distances between samples. This is set
            when creating the object to either the euclidean_distance or manhattan_distance functions defined in
            utils.py based on what argument is passed into the metric parameter of the class.

    Methods:
        fit(X, y)
            Fits the model to the provided data matrix X and targets y.

        predict(X)
            Predicts class target values for the given test data matrix X using the fitted classifier model.
    """

    def __init__(self, n_neighbors = 5, weights = 'uniform', metric = 'l2'):
        # Check if the provided arguments are valid
        if weights not in ['uniform', 'distance'] or metric not in ['l1', 'l2'] or not isinstance(n_neighbors, int):
            raise ValueError('The provided class parameter arguments are not recognized.')

        # Define and setup the attributes for the KNearestNeighbors model object
        self.n_neighbors = n_neighbors
        self.weights = weights
        self._X = None
        self._y = None
        self._distance = euclidean_distance if metric == 'l2' else manhattan_distance

    def fit(self, X, y):
        """
        Fits the model to the provided data matrix X and targets y.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.
            y: A numpy array of shape (n_samples,) representing the true class values for each sample in the input data.

        Returns:
            None.
        """
        # Intializing the training data
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        Predicts class target values for the given test data matrix X using the fitted classifier model.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the test data.

        Returns:
            A numpy array of shape (n_samples,) representing the predicted target class values for the given test data.
        """
        #calculate distance for each point of test data with each point in training data
        y_test = []
        for point in X:
            dist = []
            for i, t in enumerate(self.X_train):
                dist.append((self._distance(point, t), self.y_train[i]))
            dist1 = sorted(dist, key=lambda x:x[0])
            top_n = dist1[:self.n_neighbors]                    # take top K points
            #print("Top N: ", top_n)

            # For uniform weight, just return the label which occurred most
            if self.weights == 'uniform':
                c = Counter(ele[1] for ele in top_n)
                y_test.append(max(c, key=c.get))
            # for distance weight, take the inverse addition of distances and then return the label with max inverted distance
            else:
                d = dict()
                #print("dict for distance: ", d)
                for (dist, label) in top_n:
                    if dist == 0.0:
                        dist = 10**-7
                    if label not in d:
                        d[label] = 1 / dist
                    else:
                        d[label] += 1 / dist
                y_test.append(max(d, key=d.get))

        return y_test

