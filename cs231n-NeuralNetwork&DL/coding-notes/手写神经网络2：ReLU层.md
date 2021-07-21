# 手写神经网络2：ReLU层

> 这个系列是我在完成斯坦福公开课CS231N的作业时候所做的一些记录，这门课程是公认的深度学习入门的神课，其作业也非常硬核，需要从底层实现各种神经网络模型，包括前馈神经网络、卷积神经网络和循环神经网络，LSTM等等。
>
> 作业不仅要求掌握numpy等python库的api用法，更要对神经网络的数学理论和公式推导有非常深的理解，实现起来难度比较大，因此我也将在这个系列的博客中记录自己推导和coding的过程

ReLU层是CNN中非常常用的激活层，ReLU函数的定义为：
$$
\mathrm{ReLU}(x)=\max (0, x)
$$
我们知道全连接层进行的运算都是线性的变化，激活层的目的是就**赋予这些计算的结果非线性的特征**，使得神经网络可以拟合某个特定的目标函数，有理论研究表明，只要神经网络中采用了具有“挤压”性质的激活函数，神经网络在层数足够的情况下可以拟合任何函数到任意精度。其他常见的激活函数还有Sigmoid函数和tanh函数

## 1.1前向传播

ReLU层的前向传播比较简单，主要就是对上一层输入的x进行ReLU函数的运算，具体的代码可以用一个函数`relu_forward`来表示

```python
def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).
    Input:
    - x: Inputs, of any shape
    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    
    out = np.maximum(x, 0)
    cache = x
    
    return out, cache
```

## 1.2反向传播

反向传播的时候我们还是假设从最顶层流到当前ReLU层的关于ReLU函数的梯度是dout，我们只需要在这里层求出ReLU函数关于x的导数就可以，而ReLU函数是一个分段函数，一段是常数0因此梯度就为0，一段是线性的x因此梯度就是1，因此ReLU函数关于x的梯度就是大于0的时候为1，否则为0，因此反向传播的时候用`relu_backward`函数来计算dx：

```python
def relu_backward(dout, cache):
    """Computes the backward pass for a layer of rectified linear units (ReLUs).
    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout
    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    x[x >= 0] = 1
    x[x < 0] = 0
    dx = x * dout
    return dx
```

