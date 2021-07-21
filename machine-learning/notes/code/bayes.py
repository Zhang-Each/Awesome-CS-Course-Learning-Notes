# 似然的计算，可以直接基于似然进行判别和决策
def likelihood(x):
    """
    LIKELIHOOD Different Class Feature Likelihood
    INPUT:  x, features of different class, C-By-N numpy array
            C is the number of classes, N is the number of different feature

    OUTPUT: l,  likelihood of each feature(from smallest feature to biggest feature) 
        given by each class, C-By-N numpy array
    """
    C, N = x.shape
    l = np.zeros((C, N))
    # 这里其实给出的样本x的结构是每种分类下面不同特征属性值的分布情况，因此可以先求出每种类别的样本和
    # 再计算得到每种特征属性值对应的分布情况就可以
    class_sum = np.sum(x, axis=1)
    for i in range(C):
        for j in range(N):
            l[i, j] = x[i, j] / class_sum[i]

    return l