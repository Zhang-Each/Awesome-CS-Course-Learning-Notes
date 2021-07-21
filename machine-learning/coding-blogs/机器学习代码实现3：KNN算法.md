# 机器学习代码实现3：KNN算法

> 这一专题的内容主要是我在进行统计机器学习的算法的代码实践的过程中的一些记录，包括公式的推导，算法的实现与数据集的实验结果。
>
> 这部分工作我比较早之前就已经开始做了，现在刚好趁此机会整理一下之前写过的所有小demo，汇总成一个完成的库

## 1.KNN算法

- KNN算法是一种常见的监督学习算法，常用于分类问题，KNN算法的核心思想是用跟**输入数据最接近**的K个点的类别来预测输入数据的类别，这就决定了这个算法的关键在于：
  - 如何定义最接近？需要计算点与点之间的距离，我们一般会用L2距离(也叫欧几里德距离)来表示点与点之间的距离
  - K如何选定：K的大小会直接影响算法求解的效率
  - 怎么用K个数据的类来预测输入数据的类：一般采用投票法，选择K个点的类别中出现次数最多的类别作为预测结果
- KNN算法的过程：
  - 训练阶段：保存训练集中的数据点和对应的标签
  - 预测阶段：输入一个点之后，计算其与所有的训练集中的数据点的具体，按大小排序选出最小的K个并统计其类别的分布情况，选择出现次数的那个类别作为预测的结果

## 2.代码实现

- 这里的代码框架取自CS231N的assignment1的KNN，首先我们可以实现一种最naive的实现方式，通过**两层循环**来实现距离的计算：

```python
def compute_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.
        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.
        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                dists[i, j] = np.sqrt(np.sum((X[i] - self.X_train[j]) ** 2))
        return dists
```

- 对于测试集中的每一个$x_i$和训练集中的每一个$t_i$，KNN需要计算每一个对应的

$$
||x_i-t_{j}||_2=\sqrt{||x_i||^2_2+||t_j||^2_2-2||x_i||_2||t_j||_2}
$$

- 上面的代码`compute_distances_two_loops`就是通过两层循环实现了这样的计算，但是python中的numpy支持**向量化的运算**来大幅度提升矩阵和向量预算的效率，我们可以利用numpy库中的**广播机制**来进行向量化的运算：

$$
||X-T||_2=\sqrt{||X||_2^2+||T||_2^2-2||XT||_2}
$$

- 向量化的KNN实现方式如下：

```python
def compute_distances_no_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.
    采用向量化的形式来计算样本之间的距离，是效率最高的计算方式
    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    dists = np.sqrt(
      np.sum(X ** 2, axis=1).reshape([num_test, 1]) +
      np.sum(self.X_train ** 2, axis=1) -
      2 * np.matmul(X, self.X_train.T)
     )
    return dists
```

- 几种不同实现方式的测试如下(使用了一个比较大的数据集来跑KNN)：

| 算法     | 效率       |
| -------- | ---------- |
| 两层循环 | 33.422726s |
| 一层循环 | 26.862319  |
| 向量化   | 0.152312   |

## 3.K的选取和可视化

- 我们采用了一个二维的随机数的数据集来对KNN算法的分类边界进行可视化，选取了K=1，10，100多种值，得到的结果如下：

<img src="static/image-20210514103706165.png" alt="k=1" style="zoom: 25%;" />

<img src="static/image-20210514103744902.png" alt="k=10" style="zoom:25%;" />![image-20210514103826512](static/image-20210514103826512.png)

<img src="static/image-20210514103826512.png" alt="k=100" style="zoom:25%;" />

- 我们发现k越大，KNN算法的分类边界就越“粗犷”，而当K比较小的时候因为采样的个数太少，很容易发生过拟合