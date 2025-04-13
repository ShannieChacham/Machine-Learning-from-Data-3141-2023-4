import numpy as np

def get_random_centroids(X, k):
    '''
    Each centroid is a point in RGB space (color) in the image. 
    This function should uniformly pick `k` centroids from the dataset.
    Input: a single image of shape `(num_pixels, 3)` and `k`, the number of centroids. 
    Notice we are flattening the image to a two dimentional array.
    Output: Randomly chosen centroids of shape `(k,3)` as a numpy array. 
    '''
    centroids = []

    random_indices = np.random.choice(X.shape[0], size=k, replace=False)
    centroids = X[random_indices, :]

    # make sure you return a numpy array
    return np.asarray(centroids).astype(np.float64) 

def lp_distance(X, centroids, p=2):
    '''
    Inputs: 
    A single image of shape (num_pixels, 3)
    The centroids (k, 3)
    The distance parameter p

    output: numpy array of shape `(k, num_pixels)` thats holds the distances of 
    all points in RGB space from all centroids
    '''
    distances = []
    k = len(centroids)

    difference = np.abs(X[np.newaxis] - centroids[:, np.newaxis])
    distances = np.sum(difference**p, axis=2)**(1/p)

    return distances

def kmeans(X, k, p ,max_iter=100):
    """
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    classes = []
    centroids = get_random_centroids(X, k)

    for _ in range(max_iter):
        distances = lp_distance(X, centroids, p)
        classes = np.argmin(distances, axis=0)
        previous_centroids = centroids.copy()
        centroids = np.array([np.mean(X[classes == i], axis=0) for i in range(k)])
        if np.all(previous_centroids == centroids):
            break
            
    return centroids, classes

def kmeans_pp(X, k, p, max_iter=100):
    """
    Your implenentation of the kmeans++ algorithm.
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    num_pixels = X.shape[0]
    centroids = np.zeros((k, X.shape[1]))
    centroids[0] = X[np.random.randint(0, num_pixels)]

    for i in range(1, k):
        min_D = np.min(lp_distance(X, centroids[:i], p), axis=0)
        squared_min_distances = min_D ** 2
        probabilities = squared_min_distances / np.sum(squared_min_distances)
        centroids[i] = X[np.random.choice(num_pixels, p=probabilities)]

    for iteration in range(max_iter):
        distances = lp_distance(X, centroids, p)
        classes = np.argmin(distances, axis=0)
        new_centroids = np.array(
            [X[classes == j].mean(axis=0) if np.any(classes == j) else centroids[j] for j in range(k)]
        )
        
        if np.allclose(new_centroids, centroids):
            break
        
        centroids = new_centroids

    return centroids, classes


