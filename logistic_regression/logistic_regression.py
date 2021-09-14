import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class LogisticRegression:
    
    def __init__(self, alpha=1):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit

        # With only two features it can be written on the form:
        # Theta0 + Theta1*x1 + Theta2*x2
        self.b = np.random.randn(1, 1)
        self.W = np.random.randn(2, 1)
        self.alpha = alpha

    def J(self, X, y):
        X_num = X.to_numpy()
        m = X_num.shape[0]
        prob = self.h(X_num)

        return - (1 / m) * np.sum(
            y * np.log(prob) + (1 - y) * np.log(1 - prob)
        )

    def gradient(self, X, y):
        X_num = X.to_numpy()
        y_num = y.to_numpy()
        m = X_num.shape[0]
        prob = self.h_parameterized(X)

        error = np.mean(prob - y_num)
        print(error)
        diff = np.array(self.alpha * error * (np.mean(X_num, axis=0)))[np.newaxis]
        self.W -= diff.T
        self.b -= self.alpha * error * self.b

    def h_parameterized(self, X):
        X_num = X.to_numpy()
        result = np.zeros(X_num.shape[0], dtype='float64')
        return X_num @ self.W + self.b

    def h(self, X):
        if isinstance(X, pd.DataFrame):
            X_num = X.to_numpy()
        else:
            X_num = X
        result = np.zeros(X_num.shape[0], dtype='float64')
        return np.sum(X_num @ self.W, axis=1) + self.b

    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats containing 
                m binary 0.0/1.0 labels
        """
        # TODO: Implement
        print(self.b)
        print(self.W)
        print()
        for _ in range(10):
            self.gradient(X, y)
        print(self.b)
        print(self.W)

    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats in the range [0, 1]
            with probability-like predictions
        """
        # TODO: Implement
        print(self.h(X).shape)
        return self.h(X)[0]
        

        
# --- Some utility functions 

def binary_accuracy(y_true, y_pred, threshold=0.5):
    """
    Computes binary classification accuracy
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        The average number of correct predictions
    """
    print(y_true.shape)
    print(y_pred.shape)
    assert y_true.shape == y_pred.shape
    y_pred_thresholded = (y_pred >= threshold).astype(float)
    correct_predictions = y_pred_thresholded == y_true 
    return correct_predictions.mean()
    

def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    """
    Computes binary cross entropy 
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        Binary cross entropy averaged over the input elements
    """

    assert y_true.shape == y_pred.shape
    y_pred = np.clip(y_pred, eps, 1 - eps)  # Avoid log(0)
    return - np.mean(
        y_true * np.log(y_pred) + 
        (1 - y_true) * (np.log(1 - y_pred))
    )


def sigmoid(x):
    """
    Applies the logistic function element-wise
    
    Hint: highly related to cross-entropy loss 
    
    Args:
        x (float or array): input to the logistic function
            the function is vectorized, so it is acceptible
            to pass an array of any shape.
    
    Returns:
        Element-wise sigmoid activations of the input 
    """
    return 1. / (1. + np.exp(-x))
