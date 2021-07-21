def kmeans(x, k):
    """
    KMEANS K-Means clustering algorithm

        Input:  x - data point features, n-by-p maxtirx.
                k - the number of clusters

        OUTPUT: idx  - cluster label
                centers - cluster centers, K-by-p matrix.
                iteration_number - cluster centers of each iteration, (iter, k, p)
                        3D matrix.
    """
    
    # begin answer
    def get_current_center(centers, k, x):
        dist = np.zeros(k)
        for j in range(k):
            dist[j] = np.linalg.norm((centers[j] - x))
        min_idx = np.argmin(dist)
        return min_idx

    def get_new_center(index, x, k):
        count = 0
        sum_x = np.zeros((1, x.shape[1]))
        for i in range(len(index)):
            if index[i] == k:
                count += 1
                sum_x += x[i]
        return sum_x / count

    N, p = x.shape
    max_iter, iteration = 2, 0
    idx = np.zeros(N, dtype=np.int32)
    centers = np.zeros((k, p))
    iteration_number = np.zeros((max_iter + 1, k, p))

    # 初始化K个不同的center，随机选择数据集中的点作为中心
    first_centers = np.random.randint(0, N, k)
    for i in range(k):
        centers[i] = x[first_centers[i]]
    iteration_number[0] = centers
    while iteration < max_iter:
        # 开始迭代，每次先根据中心求出点与中心的距离，然后选择最小的点
        for i in range(N):
            idx[i] = get_current_center(iteration_number[iteration], k, x[i])
        for i in range(k):
            res = get_new_center(idx, x, i)
            centers[i] = res
        iteration += 1
        iteration_number[iteration] = centers

    # end answer

    return idx, centers, iteration_number