def PCA(data):
    """
    PCA	Principal Component Analysis

    Input:
      data      - Data numpy array. Each row vector of fea is a data point.
    Output:
      eigvector - Each column is an embedding function, for a new
                  data point (row vector) x,  y = x*eigvector
                  will be the embedding result of x.
      eigvalue  - The sorted eigvalue of PCA eigen-problem.
    """

    # Hint: you may need to **normalize** the data before applying PCA
    # begin answer
    p, N = data.shape
    normal_data = data - np.average(data, axis=1).reshape(p, 1)
    conv = np.matmul(normal_data, normal_data.T) / N
    eigen_values, eigen_vectors = np.linalg.eig(conv)
    index = np.argsort(eigen_values)[:: -1]
    eigen_values = eigen_values[index]
    eigen_vectors = eigen_vectors[:, index]
    # end answer
    return eigen_vectors, eigen_values