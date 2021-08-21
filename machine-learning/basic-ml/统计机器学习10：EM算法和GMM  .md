# 统计机器学习10：EM算法和GMM

EM算法的全称是Expectation Maximization，是用于**对含有隐变量的概率模型进行参数极大似然估计的一种迭代算法**，隐变量就是数据集的特征中中没给出但是会影响预测结果的变量。

如果没有隐变量，那么在概率模型的估计中直接进行极大似然估计，然后求导即可，而有隐变量存在的时候导数不能直接求出，所以才需要使用EM算法。EM算法主要分成E和M两个主要步骤：

-   E步骤：求解参数估计值的期望

-   M步骤：使用极大似然法求出期望的最值，然后将得到的结果放到下一轮迭代中去

EM算法的推导
------------

如果说一个概率模型中中含有可观测隐变量Z，那么我们在极大化观测数据Y关于模型参数的似然函数的时候，实际上的目标就变成了
$$
\begin{aligned}
        L(\theta)&=\log P(Y|\theta)=\log \sum_{Z}P(Y,Z|\theta)\\ &=\log \sum_{Z}P(Y|Z,\theta)P(Z|\theta)
\end{aligned}
$$
而隐变量是没有观察到的，因此不能直接使用极大似然法进行参数估计，EM算法要解决的就是在隐变量条件下的参数估计问题，两个步骤的具体过程如下：

### EM算法的输入输出要求

一般来说用Y表示观测随机变量的数据，用Z来表示隐变量的数据，Y和Z联合在一起一般称为完全数据，观测数据Y又叫做不完全数据，EM算法的求解需要知道观测变量数据Y，隐变量数据Z，联合概率分布$P(Y,Z|\theta)$，条件分布$P(Z|Y,\theta)$，最终输出的是模型的参数$\theta$

### Q函数

完全数据的对树似然函数$\log P(Y,Z|\theta)$关于在给定观测数据Y和当前参数估计$\theta_i$下对于未观测数据Z的条件概率分布$P(Z|Y,\theta)$的期望称为Q函数，也就是：
$$
Q(\theta,\theta_i)=E_Z[\log P(Y,Z|\theta)|Y,\theta_i]
$$


### EM算法求解过程

在选定好初始化参数$\theta_0$之后，需要进行：

1. E步骤：假设第i次迭代的时候参数为$\theta_i$，则在第i+1次的E步骤，计算Q函数：
   $$
   \begin{aligned}
               Q(\theta,\theta_i)&=E_Z[\log P(Y,Z|\theta)|Y,\theta_i]\\
               &=\sum_{Z}\log P(Y,Z|\theta)P(Z|Y,\theta_i)
   \end{aligned}
   $$
   

2. M步骤：求使得$Q(\theta,\theta_i)$最大化的$\theta$作为本次迭代的新估计值：
   $$
   \theta_{i+1}=\arg\max Q(\theta,\theta_i)
   $$
   

3. 重复E步骤和M步骤直到得到的参数收敛

### EM算法的有效性和收敛性证明

这部分内容《统计学习方法》上有比较详细的证明，奈何暂时看不懂，就先不管了，先理解EM算法的基本步骤再说。

高斯混合模型GMM
---------------

高斯混合模型是指如下形式的概率分布模型：
$$
P(y|\theta)=\sum_{k=1}^K\alpha_k\phi(y|\theta_k)
$$
其中$\alpha_i$表示权重系数，$\phi(y|\theta_k),\theta_k=(\mu_k,\sigma^2_k)$表示一个高斯分布的密度函数，即：
$$
\phi(y|\theta_k)=\frac{1}{\sqrt{2\pi }\sigma_k}\exp\left(-\frac{(y_k-\mu_k)^2}{2\sigma^2_k}\right)
$$


### GMM的隐变量分析

高斯混合模型中的模型参数有:$\theta=(\alpha_a,\dotsm\alpha_K,\theta_1,\dots,\theta_K)$，我们需要估计每个高斯分量的权重和自身的模型参数，而GMM中是有隐变量的，我们可以这样来分析：首先根据每个不同的权重$\alpha_k$来选择第k个分量计算生成的观测数据$y_j$，这个时候的观测结果是已知的，但是反应观测数据来自第k个分量的数据是未知的，因此我们可以用一个隐变量$\gamma_{jk}$来表示第j个观测来自于第k的模型，因此该隐变量只能取0和1两个值。

这里很重要的一点，也是非常容易出现的误区就是，观测数据的生成并不是在K个高斯模型中分别生成然后按照权重比例组合，而是以权重比例为概率分布情况，随机选择出一个高斯模型来生成观测数据$y_j$，因此隐变量$\gamma_{jk}(k=1,2,\dots,k)$中有且仅有一个是1，剩下的都是0

### GMM的求解和参数估计

EM求解GMM需要先输入N个观测数据，并输出一个高斯混合模型的参数，训练的过程主要分为E步骤和M步骤：

-   E步骤：依据当前的模型参数，计算分模型k对观测数据的响应度，即：

$$
\hat \gamma_{jk}=\frac{\alpha_k\phi(y_j|\theta_k)}{\sum\limits_{k=1}^K\alpha_k\phi(y_j|\theta_k)}
$$



-   M步骤：根据隐变量计算出新一轮迭代所需要的模型参数，模型就根据这些参数产生：

$$
\mu_k=\frac{\sum\limits_{j=1}^N\hat\gamma_{jk}y_j}{\sum\limits_{j=1}^N\gamma_{jk}}
$$

$$
\sigma_k^2=\frac{\sum\limits_{j=1}^N\hat \gamma_{jk}(y_j-\mu_k)^2}{\sum\limits_{j=1}^N\gamma_{jk}}
$$

$$
\alpha_k=\frac{\sum\limits_{j=1}^N\hat \gamma_{jk}}{N}
$$

其实这一部分一开始我没怎么看懂，后面有空继续看。