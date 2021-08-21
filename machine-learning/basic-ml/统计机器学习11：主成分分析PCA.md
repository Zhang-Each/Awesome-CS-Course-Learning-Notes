# 统计机器学习11：主成分分析PCA

基本概念
--------

主成分分析(Principal component analysis, PCA)是一种常见的无监督学习方式，**利用正交变换把线性相关变量表示的观测数据转换成几个线性无关变量所表示的数据**

-   这些线性无关的变量叫做主成分

-   主成分分析属于降维方法的一种，关于这个，统计学习方法中举了很多例子

在数据总体上进行的主成分分析叫做总体主成分分析，在有限的样本数据上进行的主成分分析称为样本主成分分析

样本主成分分析
--------------

问题的情景：对m维的随机变量x进行n次独立的观测，得到一个大小为$m\times n$的样本矩阵X，并且可以估计这些样本的均值向量：
$$
\bar x=\frac 1n \sum_{i=1}^nx_i
$$


### 样本的统计量

对于上面的样本，协方差矩阵可以表示为：
$$
s_{ij}=\frac 1{n-1}\sum_{k=1}^n(x_{ik}-\bar x_i)(x_{jk}-\bar x_j)
$$
而样本的相关矩阵可以写成$R=[r_{ij}]_{m\times m}$：
$$
r_{ij}=\frac{s_{ij}}{\sqrt {s_{ii}s_{jj}}}
$$


### 主成分的定义

我们可以设一个m维的随机变量x到m维的随机变量y的一个线性变换：
$$
\mathcal Y=(y_1,y_2,\dots,y_m)=A^Tx=\sum_{i=1}^m\alpha_ix_i
$$

$$
y_i=\alpha_i^Tx=\sum_{j=1}^m\alpha_{ji}x_j
$$



### 主成分的统计量

对于随机变量$\mathcal Y=(y_1,y_2,\dots,y_m)$，其统计量有：
$$
\begin{aligned}
        \bar y_i=\frac 1n\sum_{j=1}^n\alpha_i^Tx_j=\alpha_i \bar x \\ \mathrm{var}(y_i)=\alpha_i^T S\alpha_i \\ \mathrm{cov}(y_i,y_j)=\alpha_i^T S\alpha_j
    \end{aligned}
$$


### PCA的具体算法步骤

- 先将样本矩阵进行normalization：
  $$
  x_{ij}^*=\frac{x_{ij}-\bar x_i}{\sqrt{s_{ij}}}
  $$

- 计算样本的协方差矩阵$XX^T$并对其进行特征值的分解

- 取最大的d个特征值所对应的特征向量$w_j$

- 输出投影矩阵$