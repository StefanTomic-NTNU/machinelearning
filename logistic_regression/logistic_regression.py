import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class LogisticRegression:
    
    def __init__(self, alpha=1, max_iterations=500, circle=False):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit

        # With two features:
        # Theta0 + Theta1*x1 + Theta2*x2
        # can be written on the form:
        # b + W[0]*x1 + W[1]*x2
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.circle = circle
        self.b = np.random.randn(1, 1)
        if self.circle:
            self.W = np.random.randn(1, 1)
        else:
            self.W = np.random.randn(2, 1)

    def gradient_descent(self, X, y):
        X_num = X.to_numpy()
        y_num = y.to_numpy()
        m = X_num.shape[0]
        prob = self.h(X)
        self.W -= (self.alpha / m) * np.dot(X_num.T, (prob - y_num).T)
        self.b -= (self.alpha / m) * np.sum(prob - y_num)

    def normalize(self, X):
        X_num = X.to_numpy()
        X_0_mean = np.mean(X_num[:, 0])
        X_1_mean = np.mean(X_num[:, 1])
        X_normalized = np.copy(X)
        X_normalized[:, 0] -= X_0_mean
        X_normalized[:, 1] -= X_1_mean
        X_max = np.amax(X_normalized, axis=0)
        X_min = np.amin(X_normalized, axis=0)
        self.displacement = np.array([X_0_mean, X_1_mean], dtype='float')
        self.ratio = X_max - X_min
        X_normalized[:] /= self.ratio
        return pd.DataFrame(X_normalized, columns=['x0', 'x1'])

    def h(self, X):
        if isinstance(X, pd.DataFrame):
            X_num = X.to_numpy()
        else:
            X_num = X
        return np.sum(X_num @ self.W, axis=1) + self.b

    def preprocess_data(self, X):
        if not self.circle:
            return X
        else:
            X_preprocessed = self.normalize(X).to_numpy()
            X_preprocessed = np.sqrt(np.sum(np.square(X_preprocessed), axis=1))
            X_preprocessed = pd.DataFrame(X_preprocessed, columns=['x0'])
        return X_preprocessed

    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats containing 
                m binary 0.0/1.0 labels
        """
        X_preprocessed = self.preprocess_data(X)
        for _ in range(self.max_iterations):
            self.gradient_descent(X_preprocessed, y)

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
        return self.h(self.preprocess_data(X))[0]
        

        
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
