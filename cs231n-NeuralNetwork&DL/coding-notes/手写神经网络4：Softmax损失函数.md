# 手写神经网络3：Softmax损失函数

> 这个系列是我在完成斯坦福公开课CS231N的作业时候所做的一些记录，这门课程是公认的深度学习入门的神课，其作业也非常硬核，需要从底层实现各种神经网络模型，包括前馈神经网络、卷积神经网络和循环神经网络，LSTM等等。
>
> 作业不仅要求掌握numpy等python库的api用法，更要对神经网络的数学理论和公式推导有非常深的理解，实现起来难度比较大，因此我也将在这个系列的博客中记录自己推导和coding的过程

## 1.softmax函数的定义

- CNN卷积神经网络的输出层之前往往会有一个softmax层，我们假设CNN进行的分类问题有$M$个可能的类别，那么CNN的一系列计算得到的应该是一个$M$维的向量，每个维度$k$上的值对应着当前输入样本的是第$k$类的概率，而最后的预测结果就是这$M$个维度中的值最大的那个就是预测结果。

- 但我们很容易发现神经网络一层层算下来，你不能保证输出的$M$维向量的值处于0-1的范围内(因为要看成是概率)，这个时候就可以用softmax函数将其标准化成一个0-1的概率分布，softmax函数的形式如下：

$$
f(s_k)=\frac{\exp(s_k)}{\sum_{i=1}^M \exp(s_i)}
$$

- softmax函数的运算就可以保证最后将M维的向量转化成概率分布$ y=[f(s_1),\dots,f(s_M)]$

## 2.softmax loss的定义

- softmax loss是一种常见的CNN损失函数，实际上就是交叉熵的变体，其计算方式为：

$$
L = -\frac 1 N\sum_{i=1}^N \log (y_i)=-\frac 1 N\sum_{i=1}^N \frac{\exp(s_i)}{\sum_{k=1}^M \exp(s_k)}
$$

这里的$y$表示的是softmax层的输出结果，N是样本的个数，对每个样本的softmax损失求平均值就是数据集的损失函数

## 3.数值计算的稳定性

- 为了保证数值计算的稳定性，可以对softmax函数进行如下trick

$$
f(s_k)=\frac{\exp(s_k)}{\sum_{i=1}^M \exp(s_i)}=\frac{C\times \exp(s_k)}{C\times \sum_{i=1}^M \exp(s_i)}=\frac{\exp(s_k+\log C)}{\sum_{i=1}^M \exp(s_i+\log C)}
$$

这里加上一个常数$\log C$之后可以提高数值计算过程中softmax函数的稳定性，而这个数可以选为最大的那个$s_k$

## 4.正则化

- 为了防止过拟合，可以给softmax损失函数加上正则项，这里一般用L2正则项，而加上正则化项之后的损失函数就变成了：

$$
L=-\frac 1 N\sum_{i=1}^N \log (y_i)+\lambda||W||
$$

- 这里的$w$指的就是前一层全连接层的权重矩阵，因此CNN最靠近输出的几个层应该分别是全连接层--softmax层--输出层

## 5.反向传播求梯度

- 我们可以根据方向传播的算法求出最后一个全连接层的梯度更新公式，如下所示：

$$
\frac{\partial L}{\partial W_{j}}=\sum_i\frac{\partial L}{\partial L_i}\times \frac{\partial\hat L_i}{\partial s_j}\times \frac{\partial s_j}{\partial W_{j}}
$$

下面我们分别来计算每个部分的梯度，首先：
$$
\frac{\partial L}{\partial y_i}=-\frac 1N
$$
而softmax的求导需要分成两部分来讨论，当$i$和$j$不相等的时候：
$$
\frac{\partial y_i}{\partial s_j}=- y_i y_j
$$
而当$i=j$的时候梯度就变成了：
$$
\frac{\partial  y_i}{\partial s_i}=\hat y_i(1- y_i) = y_i- y_i^2
$$
实际上就是比两个下标不相等的时候多了一个$\hat y_i$，又因为$s=XW$(这里和代码保持一致)，所以
$$
\frac{\partial L}{\partial W_{j}}=-\frac 1N\sum_{i=1}^Nx_i^T(\hat y_i-y_i)+2\lambda W_j
$$

## 6.代码编写

- 最后编写出的代码就变成了如下形式

```python
def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)
    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.
    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength
    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W) #dW的大小是D*C

    scores = X.dot(W) # N * C
    num_train = X.shape[0]
    num_classes = W.shape[1]

    # Softmax Loss
    for i in range(num_train):
      f = scores[i] - np.max(scores[i]) # 提高数值计算的稳定性，f的大小是C
      softmax = np.exp(f)/np.sum(np.exp(f)) # softmax的大小是C
      loss += -np.log(softmax[y[i]]) # 对N个softmax求和
      for j in range(num_classes):
        dW[:,j] += X[i] * softmax[j] # D*1
      dW[:,y[i]] -= X[i]
    # 求一下均值，因为有N个样本
    loss /= num_train
    dW /= num_train
    # 加上正则项
    loss += reg * np.sum(W * W)
    dW += reg * 2 * W 
    return loss, dW
```

- 也可以用**向量化的形式**来编写softmax损失函数

```python
def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.
    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    scores = X.dot(W)
    scores = scores - np.max(scores, axis=1, keepdims=True)
    # Softmax Loss
    sum_exp_scores = np.exp(scores).sum(axis=1, keepdims=True)
    softmax_matrix = np.exp(scores) / sum_exp_scores
    loss = np.sum(-np.log(softmax_matrix[np.arange(num_train), y]) )
    # Weight Gradient
    softmax_matrix[np.arange(num_train),y] -= 1
    dW = X.T.dot(softmax_matrix)
    # Average
    loss /= num_train
    dW /= num_train
    # Regularization
    loss += reg * np.sum(W * W)
    dW += reg * 2 * W 
    return loss, dW

```

