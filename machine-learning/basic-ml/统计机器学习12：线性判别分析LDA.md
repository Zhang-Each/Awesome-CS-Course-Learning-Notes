# 统计机器学习12：线性判别分析LDA



线性判别分析(Linear Disciminant Analysis)是一种经典的降维方法，并且是一种线性的学习方法，LDA的基本想法就是，对于给定的训练集，我们可以设法将这些样本投影到一条直线上，并且使得同类样本点的投影尽可能接近而不同类的样本点的投影尽可能远离，这样一来我们实际上就需要训练出一条直线来进行特征空间中的分类，而对于测试集中的数据，可以将其同样投影到这条直线上面去，再根据投影点位置来确定该样本属于哪一类别。

LDA问题的定义
-------------

我们先来研究二分类问题的LDA，可以假设问题定义在数据机D上，用$X_i,\mu_i,\Sigma_i$分别表示某一类别的数据集合，均值和协方差矩阵，如果能够将数据投影到特征空间中的一条直线$\omega$上，那么两类样本的中心在直线上的投影分别是$\omega ^T \mu_0$和$\omega ^T \mu_1$，如果将两类样本的所有点都投影到这条直线上面，那么两类样本的协方差是$\omega ^T\Sigma_i\omega$

而我们的目的是希望同类别的点在直线上的投影尽可能靠近而不同类的尽可能远离。因此可以让同类别的投影点的协方差尽可能小，即一个目标可以定义成：
$$
\min \left(\omega ^T\Sigma_0\omega+\omega ^T\Sigma_1\omega\right)
$$
而同时我们也希望不同类别的数据样本点的投影尽可能远离，因此可以让类中心之间的距离尽可能大，则另一个目标可以表示为：
$$
\max ||\omega ^T \mu_0-\omega ^T \mu_1||^2_2
$$
这样一来，我们可以结合两个优化目标，定义目标函数：\
$$
\begin{aligned}
        J &=\frac{\left\|{\omega}^{{T}} {\mu}_{0}-{\omega}^{{T}} {\mu}_{1}\right\|_{2}^{2}}{{\omega}^{{T}}\left({\Sigma}_{0}+{\Sigma}_{1}\right) {\omega}} 
        =\frac{\left\|\left({\omega}^{{T}} {\mu}_{0}-{\omega}^{{T}} {\mu}_{1}\right)^{{T}}\right\|_{2}^{2}}{{\omega}^{{T}}\left({\Sigma}_{0}+{\Sigma}_{1}\right) {\omega}} \\
        &=\frac{\left\|\left({\mu}_{0}-{\mu}_{1}\right)^{{T}} {\omega}\right\|_{2}^{2}}{{\omega}^{{T}}\left({\Sigma}_{0}+{\Sigma}_{1}\right) {\omega}} 
        =\frac{\left[\left({\mu}_{0}-{\mu}_{1}\right)^{{T}} {\omega}\right]^{{T}}\left({\mu}_{0}-{\mu}_{1}\right)^{{T}} {\omega}}{{\omega}^{{T}}\left({\Sigma}_{0}+{\Sigma}_{1}\right) {\omega}} \\
        &=\frac{{\omega}^{{T}}\left({\mu}_{0}-{\mu}_{1}\right)\left({\mu}_{0}-{\mu}_{1}\right)^{{T}} {\omega}}{{\omega}^{{T}}\left({\Sigma}_{0}+{\Sigma}_{1}\right) {\omega}}
    \end{aligned}
$$


LDA的优化与求解
---------------

### 问题的改写

对于上面得到的优化目标，我们可以定义：类内散度矩阵(within-class scatter matrix)，用来表示同一个类别内的点的接近程度

$$
S_w=\Sigma_0+\Sigma_1=\sum_{x\in X_0}(x-\mu_0)(x-\mu_0)^T+\sum_{x\in X_1}(x-\mu_1)(x-\mu_1)^T
$$
类间散度矩阵(between-class scatter matrix)，用来表示不同类别内的点的远离程度

$$
S_b=(\mu_0-\mu_1)(\mu_0-\mu_1)^T
$$
这样一来上面的目标函数就可以变成：
$$
J=\frac{\omega^TS_b\omega}{\omega^TS_w\omega}
$$
这就是一种广义瑞丽商的优化形式。

### 瑞丽商Rayleigh quotient

对于$n\times n$的矩阵A，瑞丽商的定义形式是：
$$
R(A,x)=\frac{x^TAx}{x^Tx}
$$
瑞丽商这种形式的函数很重要的一个性质就是其最大值等于A的最大的特征值，而最小值等于A最小的特征值，即：
$$
\lambda_{\min}\le R(A,x)\le\lambda_{\max}
$$
可以用拉格朗日乘子法来证明这一结论，我们不妨设$x^tx=1$，这样一来根据拉格朗日乘子法，优化的目标函数可以写成：
$$
f(x)=x^TAx+\lambda(x^Tx-1)
$$
求梯度并令梯度等于0可以得到：
$$
\frac{\partial{f(x)}}{\partial{x}}=Ax-\lambda x = 0 \rightarrow Ax=\lambda x
$$
这其实就是矩阵特征值的定义形式，这里所有能够使得梯度为0的x就是A的所有特征值对应的特征向量，这样一来$x^TAx$在极值点的时候的计算结果就是A的一系列特征向量，因此该函数的最大值就是A最大的特征向量，而最小值就是A最小的特征向量。

而广义瑞丽商的定义是：
$$
R(A,B,x)=\frac{x^TAx}{x^TBx}
$$
广义瑞丽商可以变形为：
$$
R(A,B,x)=\frac{x^TAx}{x^TBx}=\frac{\hat x^T(B^{-\frac 12}AB^{-\frac 12})\hat x}{\hat x^T \hat x}
$$
同上面一样可以得到这个函数的最大值和最小值分别是矩阵$B^{-\frac 12}AB^{-\frac 12}$的最大的特征值和最小值。

### LDA的求解

根据瑞丽商的性质，我们可以知道LDA的求解就是对矩阵$S_w^{-\frac 12}S_bS_w^{-\frac 12}$进行特征值分解，最大的特征值对应的就是最大值，最小的特征值对应的就是最小值，同时又可以进行进一步的变形：
$$
\omega = S_w^{-1}(\mu_0-\mu_1)
$$
同时LDA也可以用贝叶斯决策理论来解释，并且可以证明，当两类数据满足同先验、满足高斯分布并且协方差相等的时候，LDA可以达到最优的分类结果。

高斯分布下的LDA
---------------

### 问题的定义

这一节内容从从贝叶斯决策理论来推导高斯分布下的LDA模型，如果样本满足高斯分布，我们先考虑二分类时候的情况，可以设两类样本分别满足高斯分布并且共享协方差矩阵，这样一类：
$$
\begin{aligned}
        P(x|y=c,\theta)=\mathcal{N}(x|\mu_c,\Sigma)\quad c\in \left\{0, 1\right\}
    \end{aligned}
$$
这样一来，根据贝叶斯公式，后验概率分布可以表示为：
$$
\begin{aligned}
        P(y=c|x,\theta)=&\frac{P(x|y=c,\theta)P(y=c,\theta)}{\sum _{c\in C}P(x|y=c,\theta)P(y=c,\theta)}\\
        \propto & \quad \pi_{c} \exp \left[{\mu}_{c}^{T} {\Sigma}^{-1} {x}-\frac{1}{2} {x}^{T} {\Sigma}^{-1} {x}-\frac{1}{2} {\mu}_{c}^{T} {\Sigma}^{-1} {\mu}_{c}\right] \\
        = & \exp \left[{\mu}_{c}^{T} {\Sigma}^{-1} {x}-\frac{1}{2} {\mu}_{c}^{T} {\Sigma}^{-1} {\mu}_{c}+\log \pi_{c}\right] \exp \left[-\frac{1}{2} {x}^{T} {\Sigma}^{-1} {x}\right]
    \end{aligned}
$$
这里我们可以令： 
$$
\begin{array}{l}
        \gamma_{c}=-\frac{1}{2} {\mu}_{c}^{T} {\Sigma}^{-1} {\mu}_{c}+\log \pi_{c} \\
        {\beta}_{c}={\Sigma}^{-1} {\mu}_{c}
    \end{array}
$$
这样一来要求的后验概率就可以写成：
$$
p(y=c \mid {x}, {\theta})=\frac{e^{{\beta}_{c}^{T} {x}+\gamma_{c}}}{\sum_{c^{\prime}} e^{{\beta}_{c^{\prime}}^{T} {x}+\gamma_{c^{\prime}}}}=\mathcal{S}({\eta})_{c}
$$
这其实就是一个softmax函数的形式，softmax函数可以把一个分布标准化成一个和为1的概率分布，而**sigmoid函数其实就是一个一元形式的softmax函数**



### 二分类下的特殊情况

对于二分类问题，我们可以进行这样的变形： 
$$
\begin{aligned}
        P(y=1 \mid {x}, {\theta}) &=\frac{e^{{\beta}_{1}^{T} {x}+\gamma_{1}}}{e^{{\beta}_{1}^{T} {x}+\gamma_{1}}+e^{{\beta}_{0}^{T} {x}+\gamma_{0}}} \\
        &=\frac{1}{1+e^{\left({\beta}_{0}-{\beta}_{1}\right)^{T} {x}+\left(\gamma_{0}-\gamma_{1}\right)}}=\operatorname{sigmoid}\left(\left({\beta}_{1}-{\beta}_{0}\right)^{T} {x}+\left(\gamma_{1}-\gamma_{0}\right)\right)
        \end{aligned}
$$
这样就可以将二分类情况下的高斯分布转化成一个sigmoid函数的形式，根据之前的$\beta,\gamma$的定义，我们有：
$$
\begin{aligned}
        \gamma_{1}-\gamma_{0} &=-\frac{1}{2} {\mu}_{1}^{T} {\Sigma}^{-1} {\mu}_{1}+\frac{1}{2} {\mu}_{0}^{T} {\Sigma}^{-1} {\mu}_{0}+\log \left(\pi_{1} / \pi_{0}\right) \\
        &=-\frac{1}{2}\left({\mu}_{1}-{\mu}_{0}\right)^{T} {\Sigma}^{-1}\left({\mu}_{1}+{\mu}_{0}\right)+\log \left(\pi_{1} / \pi_{0}\right)
    \end{aligned}
$$
因此我们可以定义： 
$$
\begin{aligned}
        {\omega} &={\beta}_{1}-{\beta}_{0}={\Sigma}^{-1}\left({\mu}_{1}-{\mu}_{0}\right) \\
        {x}_{0} &=\frac{1}{2}\left({\mu}_{1}+{\mu}_{0}\right)-\left({\mu}_{1}-{\mu}_{0}\right) \frac{\log \left(\pi_{1} / \pi_{0}\right)}{\left({\mu}_{1}-{\mu}_{0}\right)^{T} {\Sigma}^{-1}\left({\mu}_{1}-{\mu}_{0}\right)}
    \end{aligned}
$$
这样一来我们就可以得到(这个比较好推出)：
$$
\omega^Tx_0=-(\gamma_1-\gamma_0)
$$
因此，后验概率可以被写成：
$$
P(y=1|x,\theta)=\mathrm{sigmoid}(\omega^T(x-x_0))
$$
我们可以将$\omega^T(x-x_0)$看成是特征空间中的点$x$平移了$x_0$个单位之后投影到直线$\omega$上面，而sigmoid函数则是对投影的结果进行一个分类，看投影点是更靠近哪一类别。这个形式就是线性判别分析的形式，即将样本点投影到一条直线上，然后看距离多个类别的中心点的距离来判断样本的类别。

### 参数估计和模型优化

我们可以来考虑用极大似然法来训练一个模型，我们可以将该问题的对数似然函数定义为：
$$
\log P(\mathcal{D} \mid \boldsymbol{\theta})=\left[\sum_{i=1}^{N} \sum_{c=1}^{C} \mathbb{I}\left(y_{i}=c\right) \log \pi_{c}\right]+\sum_{c=1}^{C}\left[\sum_{i: y_{i}=c} \log \mathcal{N}\left(x \mid \boldsymbol{\mu}_{c}, \boldsymbol{\Sigma}_{c}\right)\right]
$$
这里的参数可以用样本的数据进行估计，比如$\pi_c=\frac{N_c}{N}$，$\mu_c=\frac{1}{N_c}\sum_{i\in N(c)}x_i$，$\Sigma_c=\frac 1N_c(x-\mu_c)(x-\mu_c)^T$，这些都是基于极大似然法的常见估计。