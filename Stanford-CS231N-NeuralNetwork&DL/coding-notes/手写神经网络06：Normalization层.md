# 手写神经网络6：Normalization层

> 这个系列是我在完成斯坦福公开课CS231N的作业时候所做的一些记录，这门课程是公认的深度学习入门的神课，其作业也非常硬核，需要从底层实现各种神经网络模型，包括前馈神经网络、卷积神经网络和循环神经网络，LSTM等等。
>
> 作业不仅要求掌握numpy等python库的api用法，更要对神经网络的数学理论和公式推导有非常深的理解，实现起来难度比较大，因此我也将在这个系列的博客中记录自己推导和coding的过程
>
> 限于个人水平实在有限，我尽量减少参考网上代码的次数

## 1.什么是Normalization？

- Normalization是神经网络训练中的另一种trick，中文译名是标准化或者归一化，我比较喜欢前一种翻译。最早应该是在[这篇论文](https://arxiv.org/pdf/1502.03167.pdf)中提出的，是一种提高神经网络训练效率和预测准确度的trick，有好几种不同的形式，比如批标准化(batch normalization)，层标准化(layer normalization)等等

## 2.批标准化Batch Normalization

### 2.1概念

- 批标准化在神经网络中通过一个批标准化层来实现，这个层一般都放在激活函数层之前，每次训练的时候选择一个小批量的数据集进行训练，然后将这样一个小批量的数据进行标准化处理，具体的处理包括：

- 先求出这个小批量数据集$\mathcal B$训练后的均值和方差：

$$
\mu_{\mathcal B}=\frac 1m\sum_{i=1}^m x_i
$$

$$
\sigma^2_{\mathcal B}=\frac 1m\sum_{i=1}^m(x_i-\mu_{\mathcal B})^2
$$

- 然后对每个x进行标准化：

$$
\hat x_i =\frac{x_i-\mu_{\mathcal B}}{\sqrt{\sigma^2+\epsilon}}
$$

- 然后再进行scale和shift，具体来说就是再对标准化后的$x$进行一个线性变换，得到结果$y$并输入到激活函数层

$$
y_i=\gamma \hat x+\beta=\gamma\frac{x_i-\mu_{\mathcal B}}{\sqrt{\sigma^2+\epsilon}} +\beta
$$

### 2.2实际训练中的动量优化

- 批标准化的训练规模相对于数据样本总体而言是比较小的，因此在训练过程中会有很多批次，而这个时候直接使用一个批的统计信息(均值和方差)来标准化一批数据相对于总体而言可能会引入较大的bias，因此真正的批标准化会使用`running_mean`和`running_var`来记录训练过程中所有出现过的批数据的统计信息，同时使用一个动量`momentum`来规定均值和方差更新时候的更新率，具体的更新公式如下：

$$
\mu_{running}=m\times \mu_{running}+(1-m)\times \mu_{batch}
$$

$$
\sigma^2_{running}=m\times \sigma^2_{running}+(1-m)\times \sigma^2_{batch}
$$

- 但是在测试模式的时候不用测试数据来更新`running_mean`和`running_var`，而是直接使用`running_mean`和`running_var`作为标准化时候所用的参数

### 2.3前向传播的代码实现

```python
def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.
    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift parameter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features
    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)
    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))
    out, cache = None, None
    if mode == "train":
        sample_mean = np.mean(x, axis=0)
        sample_var = np.var(x, axis=0)
        x_hat = (x - sample_mean) / np.sqrt(sample_var + eps)
        out = gamma * x_hat + beta
        cache = (x, gamma, beta, x_hat, sample_mean, sample_var, eps)
        # 更新running mean和variance
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
    elif mode == "test":
        hat_x = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * hat_x + beta
    else:
        raise ValueError('Invalid forward batch norm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache

```

### 2.4反向传播的推导

- 这一部分实际上是批标准化过程中最麻烦的一部分，~~因为我求导经常出问题~~，首先我们知道最终输出结果是关于样本、均值，方差三者的函数，即：

$$
y_i=\gamma f(x_i,\mu,\sigma^2)+\beta
$$

- 我们如果假设反向传递到这一层的梯度是`dout`，那么我们就需要求出$y_i$相对于$\gamma,\beta, x_i$三者的导数并用于梯度的更新，而相对于$\gamma,\beta$的导数是比较容易求的，主要的问题在于求出关于$x_i$的导数，我们不妨假设输出层处的损失函数是L，那么我们有：

$$
\frac{\partial L}{\partial \gamma}=\sum_{i=1}^m \left(\frac{\partial L}{\partial y_i}\times \hat x_i\right )=\sum_{i=1}^m \text{dout}_i \times x_i
$$

$$
\frac{\partial L}{\partial \beta}=\sum_{i=1}^m \left(\frac{\partial L}{\partial y_i}\right )=\sum_{i=1}^m \text{dout}_i
$$

- 下面我们重点来推导$y_i$关于$x_i$的导数，首先根据微积分中的基本知识，我们知道：

$$
\frac{\partial L}{\partial x_i}=\frac{\partial L}{\partial y_i}\times \frac{\partial y_i}{\partial x_i}=\text{dout}_i\times\gamma\times \frac{\partial \hat x_i}{\partial x_i}
$$

$$
\frac{\partial \hat x_i}{\partial x_i}=\frac{\partial \hat x_i}{\partial \mu_i}\times \frac{\partial \mu_i}{\partial x_i}+\frac{\partial \hat x_i}{\partial \sigma^2_i}\times \frac{\partial \sigma^2_i}{\partial x_i}+\frac{\partial \hat x_i}{\partial x_i}
$$

- 下面我们按部就班一个个来求导即可：

$$
\frac{\partial\mu }{\partial x_i}=\frac 1m
$$

$$
\frac{\partial \sigma^2}{\partial x_i}=\frac 2m(x_i-\mu)
$$

$$
\frac{\partial \hat x_i}{\partial x_i}=\frac {1}{\sqrt{\sigma^2+\epsilon}}
$$

$$
\frac{\partial \hat x_i}{\partial \sigma^2}=-\frac 12(x_i-\mu)(\sigma^2+\epsilon)^{-\frac 32}
$$

$$
\frac{\partial \hat x_i}{\partial \mu}=-\frac {1}{\sqrt{\sigma^2+\epsilon}}+\frac{\partial \hat x_i}{\partial \sigma^2}\times -\frac 2m(x_i-\mu)
$$

- 最终的梯度结果将上面的偏导数分别求出然后按照规则组合起来就可以

### 2.5反向传播的代码实现

- 这里我们使用函数`batchnorm_backward_alt`实现了批标准化的反向传播

```python
def batchnorm_backward_alt(dout, cache):
    dx, dgamma, dbeta = None, None, None
    x, gamma, beta, x_hat, mu, var, eps = cache
    M = x.shape[0]
    # 先求出比较简单的gamma和dbeta的梯度
    dgamma = np.sum(x_hat * dout, axis=0)
    dbeta = np.sum(dout, axis=0)
    # 求出dx，首先计算一个系数
    dx_hat = dout * gamma
    # 求出关于方差的导数
    d_var = np.sum(dx_hat * (x - mu) * -0.5 * ((var + eps) ** -1.5), axis=0)
    # 求出关于均值的导数
    d_mean = np.sum(dx_hat * -1 / np.sqrt(var + eps), axis=0) + d_var * np.mean(-2 * (x - mu), axis=0)
    # 最终的计算结果
    dx = 1 / np.sqrt(var + eps) * dx_hat + 2 * d_var / M * (x - mu) + 1 / M * d_mean
    return dx, dgamma, dbeta

```

- 要注意和公式一一对应

## 3.层标准化Layer Normalization

- 层标准化是另一种标准化的方式，其core idea是将每个样本的d维特征的值进行标准化，实际上就是换了一个维度的批标准化，代码实现也比较容易，基本和上面的代码完全一致，只不过把axis改成1表示换了一个维度，其他的内容基本一致，这里代码就不放出来了。

