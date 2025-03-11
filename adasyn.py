import numpy as np
from sklearn.neighbors import NearestNeighbors

def adasyn(X, y):
    """
    Implements the ADASYN algorithm as described in the original paper.

    Parameters:
    - X: ndarray, shape (n_samples, n_features)
        Feature matrix.
    - y: ndarray, shape (n_samples,)
        Target labels.

    Returns:
    - X_resampled: ndarray
        Resampled feature matrix with synthetic samples.
    - y_resampled: ndarray
        Resampled target labels.
    """

    # Identify minority and majority class
    classes, class_counts = np.unique(y, return_counts=True)
    minority_class = classes[np.argmin(class_counts)]
    majority_class = classes[np.argmax(class_counts)]
    
    X_minority = X[y == minority_class]
    n_minority = len(X_minority)
    n_majority = len(X[y == majority_class])

    # Step 1: Compute the number of synthetic samples to generate
    d = n_majority - n_minority  # Imbalance factor
    G = d  # Total synthetic samples needed

    # Step 2: Find k-nearest neighbors for each minority sample
    k = 5
    knn = NearestNeighbors(n_neighbors=k+1).fit(X)
    neighbors = knn.kneighbors(X_minority, return_distance=False)[:, 1:]

    # Step 3: Compute the imbalance degree ri for each minority sample
    ri = np.array([sum(y[neighbors[i]] != minority_class) / k for i in range(n_minority)])
    if ri.sum() == 0:
        return X, y  # No synthetic samples needed
    ri = ri / ri.sum()  # Normalize ri to sum to 1

    # Step 4: Generate synthetic samples adaptively
    X_synthetic = []
    for i in range(n_minority):
        Gi = int(G * ri[i])  # Number of samples to generate for instance i
        for _ in range(Gi):
            neighbor = np.random.choice(neighbors[i])  # Random neighbor
            gap = np.random.rand()  # Random interpolation factor
            synthetic_sample = X_minority[i] + gap * (X[neighbor] - X_minority[i])
            X_synthetic.append(synthetic_sample)

    X_synthetic = np.array(X_synthetic)
    y_synthetic = np.full(len(X_synthetic), minority_class)

    # Step 5: Return the augmented dataset
    X_resampled = np.vstack((X, X_synthetic))
    y_resampled = np.hstack((y, y_synthetic))

    return X_resampled, y_resampled