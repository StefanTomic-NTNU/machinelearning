import numpy as np 
import pandas as pd
import random as r
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)





class KMeans:
    
    def __init__(self, k=2, normalize_points=True, k_means_plus_plus=True):
        # NOTE: Feel free add any hyperparameters
        # (with defaults) as you see fit
        self.k_means_plus_plus = k_means_plus_plus
        self.normalize_points = normalize_points
        self.centroids = None
        self.k = k
        self.displacement = None
        self.ratio = None

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

    def fit(self, X):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
        """
        # TODO: Implement
        n = X.shape[0]
        if self.normalize_points:
            X = self.normalize(X)
        X_num = X.to_numpy()

        if not self.k_means_plus_plus:
            self.centroids = np.zeros((self.k, 2))
            for j in range(self.k):
                index = np.random.choice(n)
                self.centroids[j, :] = X_num[index]
        else:
            # K means++
            self.centroids = np.zeros((2, 2), dtype='float64')
            index = np.random.choice(n)
            self.centroids[0, :] = X_num[index]
            index = np.random.choice(n)
            self.centroids[1, :] = X_num[index]
            # print(self.centroids)
            for _ in range(2, self.k):
                # print(self.centroids)
                distances = cross_euclidean_distance(X_num, self.centroids)
                min_index = np.argmin(distances, axis=1)
                max_index = None
                max_distance = 0
                for i in range(len(min_index)):
                    dist = distances[i, min_index[i]]
                    if dist > max_distance:
                        max_distance = dist
                        max_index = i
                self.centroids = np.vstack([self.centroids, X_num[max_index]])

        # Assigning of clusters
        for _ in range(20):
            # print(self.centroids)
            distances = cross_euclidean_distance(X_num, self.centroids)
            # print(distances)

            # centroid blir ikke assignet til noen punkter
            min_index = np.argmin(distances, axis=1)

            count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            for j in range(self.k):
                points = []
                for i in range(n):
                    if min_index[i] == j:
                        points.append(X_num[i, :])
                        count[j] += 1
                # print(points)
                # print(np.mean(np.array(points, dtype='float64'), axis=0))
                self.centroids[j, :] = np.mean(np.array(points, dtype='float64'), axis=0)
            # print(count)

        # print(self.centroids)
        if self.normalize_points:
            self.centroids *= self.ratio
            self.centroids += self.displacement


    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m integer array with cluster assignments
            for each point. E.g., if X is a 10xn matrix and 
            there are 3 clusters, then a possible assignment
            could be: array([2, 0, 0, 1, 2, 1, 1, 0, 2, 2])
        """
        # TODO: Implement
        if self.normalize_points:
            self.centroids -= self.displacement
            self.centroids /= self.ratio
            X = self.normalize(X)
        distances = cross_euclidean_distance(X.to_numpy(), self.centroids)
        if self.normalize_points:
            self.centroids *= self.ratio
            self.centroids += self.displacement
        return np.argmin(distances, axis=1)
    
    def get_centroids(self):
        """
        Returns the centroids found by the K-mean algorithm
        
        Example with m centroids in an n-dimensional space:
        >>> model.get_centroids()
        numpy.array([
            [x1_1, x1_2, ..., x1_n],
            [x2_1, x2_2, ..., x2_n],
                    .
                    .
                    .
            [xm_1, xm_2, ..., xm_n]
        ])
        """
        # TODO: Implement 
        return self.centroids
    
    
    
    
# --- Some utility functions 


def euclidean_distortion(X, z):
    """
    Computes the Euclidean K-means distortion
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the raw distortion measure 
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    distortion = 0.0
    for c in np.unique(z):
        Xc = X[z == c]
        mu = Xc.mean(axis=0)
        distortion += ((Xc - mu) ** 2).sum()
        
    return distortion

def euclidean_distance(x, y):
    """
    Computes euclidean distance between two sets of points 
    
    Note: by passing "y=0.0", it will compute the euclidean norm
    
    Args:
        x, y (array<...,n>): float tensors with pairs of 
            n-dimensional points 
            
    Returns:
        A float array of shape <...> with the pairwise distances
        of each x and y point
    """
    return np.linalg.norm(x - y, ord=2, axis=-1)

def cross_euclidean_distance(x, y=None):
    """
    Compute Euclidean distance between two sets of points 
    
    Args:
        x (array<m,d>): float tensor with pairs of 
            n-dimensional points. 
        y (array<n,d>): float tensor with pairs of 
            n-dimensional points. Uses y=x if y is not given.
            
    Returns:
        A float array of shape <m,n> with the euclidean distances
        from all the points in x to all the points in y
    """
    y = x if y is None else y 
    assert len(x.shape) >= 2
    assert len(y.shape) >= 2
    return euclidean_distance(x[..., :, None, :], y[..., None, :, :])


def euclidean_silhouette(X, z):
    """
    Computes the average Silhouette Coefficient with euclidean distance 
    
    More info:
        - https://www.sciencedirect.com/science/article/pii/0377042787901257
        - https://en.wikipedia.org/wiki/Silhouette_(clustering)
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the silhouette score
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    # Compute average distances from each x to all other clusters
    clusters = np.unique(z)
    D = np.zeros((len(X), len(clusters)))
    for i, ca in enumerate(clusters):
        for j, cb in enumerate(clusters):
            in_cluster_a = z == ca
            in_cluster_b = z == cb
            d = cross_euclidean_distance(X[in_cluster_a], X[in_cluster_b])
            div = d.shape[1] - int(i == j)
            D[in_cluster_a, j] = d.sum(axis=1) / np.clip(div, 1, None)
    
    # Intra distance
    a = D[np.arange(len(X)), z]

    # Smallest inter distance 
    inf_mask = np.where(z[:, None] == clusters[None], np.inf, 0)
    b = (D + inf_mask).min(axis=1)
    return np.mean((b - a) / np.maximum(a, b))


if __name__ == '__main__':
    k_means = KMeans()
