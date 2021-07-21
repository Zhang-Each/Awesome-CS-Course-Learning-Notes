# 1.机器学习的基本概念


## 1.1什么是Machine Learning？


​		机器学习(Machine Learning)实际上就是计算机系统的自我学习，通过输入的数据集来学习出一个"函数(或者说映射)"，机器学习的三大问题是：

-   监督学习Supervised
    Learning：学习已经存在的结构和规则，也就是学习一个映射，即对于有标注数据的学习，常见的有分类和回归

-   非监督学习Unsupervised
    Learning：学习过程中由计算机自己发现新的规则和结构，即对于无标注数据的学习，常见的有聚类和降维

-   强化学习：有反馈的学习，但是只会反馈对和错，在机器人中比较常见

监督学习的数据都是有标注的，而非监督学习的数据是没有标准的，需要在学习过程中自己发现数据中存在的一些结构和特征

### 1.1.1机器学习中最重要的问题

- 机器学习问题中最重要并需要花最多时间的事情就是定义模型，机器学习的三个要素是莫名的表示、度量和优化。

几个机器学习中的基本概念


-   Sample，example，pattern 问题案例，样本

-   Features，predictor，independent variable
    将需要处理的数据用高维向量来表示，一般用$x_i$表示

-   State of the nature，lables，pattern class
    数据的类型，一般用$\omega_i$表示

-   Training data：用若干组$(x_i, \omega_i)$表示训练数据集

-   Test data 测试数据

-   Training error & Test error 训练误差和测试误差

## 1.2机器学习问题的分类

事实上1.1中介绍的三大机器学习问题只是机器学习的一种分类方法，常见的对于机器学习的分类方法还有如下几种：

### 1.2.1按模型分类

根据模型的类型，机器学习还存在如下几种分类方式：按照是否为概率模型分类、按照是否为线性模型分类、按照是否为参数化模型分类，每种分类方式的特点如下：

-   概率模型和非概率模型：非概率模型也叫做确定性模型，区别在于概率模型在学习时使用条件概率分布形式，二非概率模型使用函数形式

-   线性模型和非线性模型：主要区别在于模型是否为线性函数，神经网络就是复杂的非线性模型

-   参数化模型和非参数化模型：区别在于参数化模型用确定个参数刻画模型，而非参数化模型的参数维度不固定

### 1.2.2按算法分类

可以分为在线学习和批量学习，在线学习是每次接受一个样本进行预测，然后进行学习，并不断重复的过程，而批量学习则是一次性把数据集训练完之后再
进行结果的预测

## 1.3模型的复杂度 Complexity
-----------------------

随着模型的复杂度提高，其训练误差会不断下降，但是测试误差会先下降再提高。因此模型存在一个最优的复杂度。模型训练的过程中可能会出现过拟合(overfitting)的情况.

按照我个人的表达方式，过拟合其实就是对测试数据拟合得太好而对训练数据拟合效果不好。
**Definition 1**.
*泛化(Generalization)能力：表示一个模型对未出现过的数据样本的预测能力，我们一般希望泛化能力越大越好。*

## 1.4模型的评估

### 1.4.1样本数据集的表示

我们用$D = \lbrace(x_1,y_1),(x_2,y_2),\dots,(x_m,y_m)\rbrace$表示样本数据集 其中 $y_i$表示 $x_i$的真实标记，要评估学习的性能，需要将预测结果和真实标记进行比较

### 1.4.2指示函数

指示函数 $\mathbb I(e)$在表达式e的值为真的时候值为1，在表达式e为假的时候值为0

### 1.4.3错误率和准确率

- 模型训练的错误率定义为 $$\begin{aligned}
      E(f; D)=\frac{1}{m}\sum_{i=1}^{m}\mathbb I(f(x_i)\not=y_i)\end{aligned}$$

- 精确度的定义为： $$\begin{aligned}
      acc(f;D)=\frac{1}{m}\sum_{i=1}^{m}\mathbb I(f(x_i)=y_i)=1-E(f;D)\end{aligned}$$

### 1.4.4查准率和查全率

  真实情况   结果为正例   结果为反例
---------- ------------ ------------
  正例       TP           FN
  反例       FP           TN

-   查准率 $P=\frac{TP}{TP+FP}$ 表示预测结果为正例中预测正确的比例

-   查全率 $R=\frac{TP}{TP+FN}$ 表示所有正例中被预测对的比例

## 1.5No Free Lunch

​		我们总希望我们的机器学习算法在所有情况下都表现得非常优秀，因为这样可以帮我们省很多事，然而事实上这是不可能的，因为对于同一个问题的两种解决算法A和B，如果A在某些情况下表现比B要好，那么A就一定会在另一些情况里表现得比B要差，这是因为对于一个问题，其所有情况的总误差和算法是没有关系的，也就是说，一个特定问题的所有可能情况的总误差也是一定的。

​		下面我们可以来简单地证明这一个结论，我们用X表示样本空间，H表示假设空间，并且假设它们都是离散的，令$P(h|D,\lambda_a)$表示算法a在训练集D下产生假设h的概率，再用f代表我们希望学习的真实目标函数，则可以用$C=X-D$来表示训练集之外的所有样本，则其产生的误差可以表示为：
$$\begin{aligned}
​    E(\lambda_a|D,f)=\sum_h\sum_{X\in C} P(x)\mathbb{I}(h(x)\not= f(x))P(h|D,\lambda_a)\end{aligned}$$
​		考虑最简单的二分类问题，并且真实目标函数可以是任何映射到0和1上的函数，因此可能的函数有$2^{|X|}$,对所有可能的f按照均匀分布求和，有
$$\begin{aligned}
​    \sum_f E(\lambda_a|D,f) & =\sum_f\sum_h\sum_{X\in C} P(x)\mathbb{I}(h(x)\not= f(x))P(h|D,\lambda_a)\\
​    & = \sum_{x\in C}  P(x)\sum_h P(h|D,\lambda_a) \sum_f\mathbb{I}(h(x)\not= f(x)) \\
​    & = \sum_{x\int C} P(x) \sum_h P(h|D,\lambda_a) \frac 12 2^{|X|}\\
​    & = 2^{|X|-1}\sum_{x\in C}P(x)\sum_h P(h|D,\lambda_a)\\
​    & = 2^{|X|-1}\sum_{x\in C}P(x)
\end{aligned}$$
我们发现这其实是一个常数，也就是说不管选择了什么算法，其在特定问题和特定数据集下的总误差是一定的，因此两个算法一定会在一些问题上的
表现互有胜负，这也就是There is no free lunch定理。

## 1.6误差，偏差和方差


在机器学习中我们非常关注学习的效果，这可以通过误差的指标来衡量，常见的一种误差就是均方误差，比如在回归问题中，均方误差可以表示为：
$$E(f; \mathcal D)=\frac{1}{m}\sum_{i=1}^{m}(f(x_i)-y_i)^2$$
如果采用概率密度函数，就可以计算连续状态下的均方误差：
$$E(f; \mathcal D)=\int_{x\thicksim D}(f(x_i)-y_i)^2p(x)dx$$
而均方误差又可以进一步的分解。


对于数据集D和学习模型f，学习算法期望预测为：
$$\overline f(x)=\mathbb E_{D}(f(x;\mathcal D))$$
则根据方差的定义，可以得到方差的表达式：
$$var(x)=\mathbb E_{D}[(f(x;\mathcal D)-\overline f(x))^2]$$
我们又可以定义**模型的期望预测值和真实标记之间的误差为偏差**(bias)，即
$$bias^2(x)=(\overline f(x)-y)^2$$
则在回归问题的均方误差中，我们可以将均方误差分解为：
$$E(f;\mathcal D)=var(x)+bias^2(x)+\epsilon^2$$
其中$\epsilon^2=\mathbb E_{D}[(y_D-y)^2]$表示样本产生的噪声(noise)


贝叶斯理论 Bayes' Theory
========================

贝叶斯理论是非常经典的分类算法，我们一般用$x$表示样本，用$\omega_j$表示可能的分类类别，
则$P(\omega_j| x)$表示$x$属于这一类别的概率。
贝叶斯决策论是在概率框架下实施决策的基本方法，本质上是如何基于概率和误判损失来选择最优的类别标记.

贝叶斯公式的推导
----------------

根据条件概率公式，我们有 $$P(AB)=P(A|B)P(B)=P(B|A)P(A)\\
    \Rightarrow P(A|B)=\frac{P(B|A)P(A)}{P(B)}$$
这就是贝叶斯公式最简单的基本形式，其中

-   $P(A)$ 是先验概率(prior),指的是样本中各种情况出现的概率

-   $P(B|A)$ 是似然(likelihood),表示A发生的条件下，B出现的概率

-   $P(A|B)$ 是后验概率(posterior)

现在我们假设有若干个特征$\omega_1,\omega_2,…,\omega_c$,对于数据集D中的一个样本x，有

$$\begin{aligned}
    P(\omega_j|x)=\frac{P(x|\omega_j)P(\omega_j)}{P(x)}\\
    P(x)=\sum_{j=1}^{c}P(x|\omega_j)P(\omega_j)  \end{aligned}$$
贝叶斯定理可以用于分类问题的决策，而用贝叶斯定理进行决策的实质是通过概率分布
情况来使得分类错误的可能性最小化，上面提到的公式事实上是基于后验概率来进行分类决策
的，也称为Optimal Bayes Decision Rule

贝叶斯决策也可能会碰到一些特殊情况，比如当先验概率相等的时候，只需要比较likelihood
就可以，当likelihood一样大小的时候只需要比较先验概率就可以。

贝叶斯公式的Loss
----------------

### 条件风险

可以定义一个loss
function来估计贝叶斯公式的loss，我们设$\lambda(\alpha_i|\omega_j)$
表示将原本类型为$\omega_j$的样本分类成了$\alpha_i$所带来的loss(也可以叫risk)，则将数据集D中的样本x
分类为$\alpha_i$所带来的条件风险(Condition Risk)可以表示为：
$$R(\alpha_i|x)=\sum\limits_{j=1}\limits^{c}\lambda(\alpha_i|\omega_j)P(\omega_j|x)$$
而对于所有可能的$\alpha_i$，可以对其条件风险进行积分，得到总的条件风险
$$R=\int R(\alpha_i|x)p(x)dx$$
可以记$\lambda_{ij}=\lambda(\alpha_i|\omega_j)$则对于一个二分类问题，我们只需要比较$R(\alpha_i|x)$的大小，将其展开之后发现只需要比较
$\frac{P(x|\omega_1)}{P(x|\omega_2)}$和$\frac{\lambda_{12}-\lambda_{22}}{\lambda_{21}-\lambda_{11}}\times \frac{P(\omega_2)}{P(\omega_1)}$的大小。

### 0-1 loss

一种简单的loss定义是损失在分类正确的时候为0，错误的时候为1，即
$$\lambda_{ij}=\left\{
\begin{aligned}
0 \quad (i=j) \\
1 \quad (i\neq j)
\end{aligned}
\right.$$ 将其带入原本的条件风险表达式，我们可以得到 $$\begin{aligned}
    R(\alpha_i|x)=\sum\limits_{j=1}\limits^{c}\lambda(\alpha_i|\omega_j)P(\omega_i|x)=\sum\limits_{j\not=i}{P(\omega_i|x)}=1-P(\omega_i|x)\end{aligned}$$
此时我们进行决策的话只需要比较$P(\omega_i|x)$的大小，而根据贝叶斯公式，我们只需要比较$P(x|\omega_i)P(\omega_i)$的大小，因此下面我们就来解决这一部分的计算问题。

### Coding体验I

在学完这一部分的内容之后我尝试了完成蔡登老师的作业，在给定的框架上实现一个最简单形式的贝叶斯二分类器，
需要编写的部分包括likelihood的计算、
posterior的计算，以及错误分类和risk的计算，这个作业主要分成了三个部分，首先是实现likelihood并基于likehood进行二分类，然后是实现
posterior并基于posterior进行分类，分别计算两种分类的误判数目并比较，然后是计算risk，其中我有如下几点收获：

-   这个作业的样本分布是离散的，并且有一定的上下界，因此可以先获取样本值的上下界，然后统计样本的分布情况，再进行后续计算

-   在计算likelihood的时候，要计算的实际上是当前分类下，此类样本的所占比例，因此每种样本下每种特征的个数除以每种分类下样本总数，就是likelihood

-   在计算posterior的时候，$P(\omega_j)$是每种类别占全部样本的比例，而$P(x)$是当前特征属性值x的样本的likelihood和先验概率的乘积之和，也就是说$P(x)$其实是多个值，并不是一整个值，而是该特征的每一种值都有一个$P(x)$

-   在计算误判个数的时候要先根据分类依据确定每种特征属性值的最后分类结果，然后遍历所有的测试集找出分类错误的

-   在计算risk的时候，也是每种特征属性的值都有一个对应的risk

参数估计 Parameter Estimation
-----

经过刚才的推导我们发现，最后只需要计算$P(x|\omega_i)P(\omega_i)$就可以进行贝叶斯决策，而$P(\omega_i)$是可以直接在样本中计算出来的，因为监督学习中每个
样本都是有label的，可以非常容易地计算出$P(\omega_i)$，问题就在于如何计算$P(x|\omega_i)$，也就是类别$\omega_i$中出现的样本值为x的概率。

我们可以用数据集D的样本来对$P(x|\omega_i)$进行估计，而估计的主要方法有极大似然法(Maximum-Likelihood)和贝叶斯估计法两种。

### 正态分布 Normal Distribution

我们需要先回忆一下概率论中学过的正态分布的相关知识，因为后面极大似然估计中会用到。正态分布也叫做高斯分布，很明显这个分布是数学王子高斯
发现的，正态分布的形式如下：\
对于一维变量我们有：
$$P(x|\mu,\sigma^2)=\mathcal{N}(x|\mu,\sigma^2)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp\lbrace -\frac{(x-\mu)^2}{2\sigma^2}\rbrace$$
并且$E(x)=\mu,var(x)=\sigma^2$,而对于d维的向量x，多元高斯分布的参数是d维的均值向量$\mu$和$d\times d$的对称正定协方差矩阵$\Sigma$
$$P(\boldsymbol x|\mu,\Sigma)=\mathcal{N}(x|\mu,\Sigma^2)=\frac{1}{({2\pi})^{\frac{d}{2}}|\Sigma|^{\frac{1}{2}}}\exp[-\frac{1}{2}(\boldsymbol x-\mu)^{T}\Sigma^{-1}(\boldsymbol x-\mu)]$$

### 极大似然法

估计模型参数的时候最常用的方法就是极大似然估计，对于一个包含m个样本的数据集X，我们可以假设它是由一个概率分布$p_{\mathrm{data}}(\mathbf x)$生成的，现在我们要用这个数据集来估计出一个$p_{\mathrm{model}}(\mathbf x)$来近似地比噢傲视真实的概率模型，我们可以给模型设定一族参数$\theta$，因此模型可以表示为$p_{\mathrm{model}}(\mathbf x,\theta)$，这样一来，极大似然法可以定义成：
$$\theta_{\mathrm{ML}}=\arg\max\limits_{\theta}p_{\mathrm{model}}(\mathbf x,\theta)=\arg\max\limits_{\theta}\prod _{i=1}^m p_{\mathrm{model}}(x^{(i)},\theta)$$
但是多个概率的积不太容易计算，因此可以给目前函数取一个对数：
$$\theta_{\mathrm{ML}}=\arg\max\limits_{\theta}\sum _{i=1}^m \log p_{\mathrm{model}}(x^{(i)},\theta)$$
因为重新缩放代价函数的时候argmax不会改变，因此可以除以样本的大小m得到和训练数据经验分布相关的期望作为极大似然法的评价准则：
$$\theta_{\mathrm{ML}}=\arg\max\mathbb E_{\mathbf x -p_{\mathrm {data}}}[\log p_{\mathrm{model}}(x,\theta)]$$
一种解释最大似然估计的观点就是**将极大似然法看作最小化训练集上的经验分布和模型分布之间的差异**，而两者之间的差异可以根据KL散度进行度量，即：
$$D_{KL}(\hat p_{\mathrm{data}}||p_{\mathrm{model}})=\mathbb E_{\mathbf x -p_{\mathrm {data}}}[\log \hat p_{\mathrm{data}}(x) - \log p_{\mathrm{model}}(x)]$$

### 贝叶斯定理的参数估计

我们假定需要进行分类的变量在某一类别下其样本的值是服从高斯分布的，则有
$$P(\omega_i|x)=P(x|\omega_i)P(\omega_i)=P(x|\omega_i,\theta_i)P(\omega_i), \theta_i=(\mu_i,\sigma_i)$$
其中$\theta_i$为待估计的参数。我们定义整个数据集D中类别为$\omega_i$的子集是$D_i$,其对于参数$\theta_i$的似然为
$$P(D_i|\theta)=\prod\limits_{x_k\in D_i}P(x_k|\theta_i)$$
极大似然法的基本思路就是让这个数据集的似然达到最大，而达到最大的时候的参数值就是我们要求的参数估计值，因为它使得数据集中
可能属于这个类别的样本的概率达到了最大。而为了防止数值过小造成**下溢**，可以采用对数似然
$$l(\theta) = \ln P(D_i|\theta) = \sum\limits_{x_k\in D_i}\ln P(x_k|\theta_i)$$
我们的目标就是$\theta^*=\arg\max\limits_{\theta} l(\theta)$

我们之前已经假设了某一类别下的x服从正态分布，则有 $$\begin{aligned}
        \ln P(x_k|\theta_i) & = \ln (\frac{1}{({2\pi})^{\frac{d}{2}}|\Sigma|^{\frac{1}{2}}}\exp[-\frac{1}{2}(x_k-\mu_i)^{T}\Sigma^{-1}(x_k-\mu_i)])\\
                            & = -\frac{d}{2}\ln (2\pi)-\frac{1}{2}\ln |\Sigma|-\frac{1}2 (x_k-\mu_i)^{T}\Sigma^{-1}(x_k-\mu_i)
    \end{aligned}$$ 则对$\mu_i$求偏导数得到 $$\begin{aligned}
    \frac{\partial \ln P(x_k|\theta_i)}{\partial \mu_i}=\Sigma^{-1}(x_k-\mu_i)\end{aligned}$$
我们需要让对数似然函数取得最值，则对其求偏导数可得到 $$\begin{aligned}
    \sum_{x_k\in D_i}\Sigma^{-1}(x_k-\mu_i)=0\Rightarrow \mu_i=\frac{1}{n}\sum_{x_k\in D_i} x_k\end{aligned}$$
同理可以对$\Sigma_i$进行求导可以得到 $$\begin{aligned}
    \frac{\partial \ln P(x_k|\theta_i)}{\partial \Sigma_i}=\frac{1}{2\Sigma}+\frac{1}{2\Sigma^2}(x_k-\mu_i)(x_k-\mu_i)^T\end{aligned}$$
因此可以求得$\Sigma$的估计值 $$\begin{aligned}
    \Sigma^2=\sum_{x_k\in D_i}(x_k-\mu_i)(x_k-\mu_i)^T=\sum_{x_k\in D_i}||x_k-\mu_i||^2\end{aligned}$$
这些参数按照上面的估计公式计算之后可以带入原本的likelihood表达式计算出likelihood，进一步计算出posterior

### 代码实现

``` {.python language="python"}
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
```

### 贝叶斯估计

极大似然法是频率学派的方法，而贝叶斯估计则是贝叶斯派的估计方法，区别在于极大似然法MLE认为估计的参数是一个fixed
value但是贝叶斯派则认为 它是随机的变量. 把训练集D作为变量，则有
$$P(\omega_i|x,D)=\frac{P(x|\omega_i,D)P(\omega_i,D)}{\sum P(x|\omega_i,D)P(\omega_i,D)}$$
又可以化简为
$$P(\omega_i|x,D)=\frac{P(x|\omega_i,D_i)P(\omega_i)}{\sum P(x|\omega_i,D_i)P(\omega_i)}$$

朴素贝叶斯 Naive Bayes
----------------------

朴素贝叶斯分类器的基本思想是，既然我们的困难是$P(x|\omega_j)$涉及到x所有属性的联合概率不好估计，那我们就把联合概率的计算难度降到最低，
也就是假设x的所有属性(也可以叫做特征)是互相独立的，此时对于d维的样本$x\in D$,贝叶斯的公式变成了
$$\begin{aligned}
    P(\omega|x)=\frac{P(\omega)P(x|\omega)}{P(x)}=\frac{P(\omega)}{P(x)}\prod_{i=1}^dP(x_i|\omega)\end{aligned}$$
类似地，对于所有类别来说P(x)是相同的，因此朴素贝叶斯分类的目标就是
$$h_{nb}(x)=\arg\max_{c\in Y}P(\omega)\prod_{i=1}^{d}P(x_i|\omega)$$
训练集D中，令$D_c$表示第c类样本构成的集合，则类的先验概率
$$\begin{aligned}
    P(c)=\frac{|D_c|}{|D|}\end{aligned}$$
对于离散的属性而言，可以令$D_{c,x_i}$
表示$D_c$中第i个特征的取值为$x_i$的样本组成的集合，则条件概率可以估计为
$$\begin{aligned}
    P(x_i|\omega)=\frac{|D_{c,x_i}|}{|D_c|}\end{aligned}$$

::: {.definition}
**Definition 2**.
*拉普拉斯修正$Laplas Smoothing$：样本存在局限性，不可能所有的特征都恰好在样本中出现，特别是朴素贝叶斯的完全独立假设使得样本可能的特征数量变得特别
多，我们可以假定所有的特征大致上都是均匀分布的，通过在训练集上添加K个特征(K种不同的特征，每种类型各一个)使得每种特征都可以出现，此时的先验概率估算公式
变成了：*
:::

$$P(x_i|\omega)=\frac{|D_{c,x_i}|+1}{|D_c|+1}$$

线性回归 Linear Regression
==========================

前面说到机器学习中的任务主要是分类和回归，分类如前面的贝叶斯定理所展现的那样是输出一个类别(有限的离散数值)，而回归的任务
则是输出一个real value

线性回归的基本思路
------------------

对于n维的样本$\boldsymbol x=[x_1,x_2,…,x_n]$
找到一系列线性函数$\omega=[\omega_1,\omega_2,…,\omega_n]$
使得$f(x)=\omega^T\boldsymbol x+b$，
这个问题可以变形为n+1维使得$f(x)=\omega^T\boldsymbol x$
其中$\boldsymbol x=[1,x_1,x_2,…,x_n]$而$\omega=[b,\omega_1,\omega_2,…,\omega_n]$，这样以来
表达式的形式更加统一了，而我们的目标就是训练出这个函数f，使得对于训练数据$(x_i,y_i)$，学习出一个函数f使得$f(x_i)=y_i$，我们要求解
的对象就是这个n+1维的向量$\omega$

线性回归的loss函数和解
----------------------

线性回归常见的loss函数定义是最小平方误差(MSE)
$$MSE=\frac{1}{n}\sum\limits_{i=1}\limits^{n}(y_i-f(x_i,\omega))^2$$
也会使用残差平方和(RSS)，其形式如下：
$$J_n(\alpha)=\sum\limits_{i=1}\limits^{n}(y_i-f(x_i,\omega))^2=(\boldsymbol y-X^T\omega)^T(\boldsymbol y-X^T\omega)$$
我们对RSS的表达式求梯度有：
$$\nabla J_n(\alpha)=-2X(\boldsymbol y-X^T\alpha)=0$$
因此可以得到使得RSS最小的线性回归的解是: $$\omega=(XX^T)^{-1}Xy$$
我们需要注意到，每一个样本x是$d\times 1$维的向量，因此X是$d\times n$维的矩阵，而y也是$n\times 1$维的向量，所以当样本数n小于特征数d的时候，
$XX^T$是不满秩的，求不出逆矩阵，此时的线性回归有多个解，其实这也很好理解，因为要求解的是n+1维的向量$\omega$，有n+1个变量，因此至少
需要n+1个样本才能确定n+1个参数，出现上述情况的时候所有可能的解都可以使得均方误差最小化，此时可以考虑引入正则化项来筛选出需要的结果。

线性回归的统计模型
------------------

真实情况下的数据样本往往会有噪声，比如$y=f(\boldsymbol x,\omega)+\epsilon$
其中$\epsilon$是一个随机噪声，服从$N(0,\sigma^2)$的正态分布,
此时可以通过极大似然法来估计$\omega$，定义
$$P(y|\boldsymbol x,\omega,\sigma)=\frac{1}{\sqrt{2\pi}\sigma}\exp[-\frac{1}{2\sigma^2}(y-f(\boldsymbol x,\omega))^2]$$
根据极大似然法有
$$L(D,\omega,\sigma)=\prod_{i=1}\limits^nP(y_i|\boldsymbol x_i,\omega,\sigma)$$
我们的求解目标变成了：
$$\omega=\arg\max L(D,\omega,\sigma)= \arg \max\prod_{i=1}\limits^nP(y_i|\boldsymbol x_i,\omega,\sigma)$$
取对数似然之后有
$$l(D,\omega,\sigma)=-\frac{1}{2\sigma^2}\sum_{i=1}\limits^{n}(y_i-f(x_i,\omega)^2)+c(\sigma)$$
到这一步为止我们又回到了RSS，因此解的表达式依然和上面推出的是一样的。

岭回归 Ridge Regression
-----------------------

我们发现普通的线性回归很容易出现overfitting的情况，比如一些系数$w_i$的值非常极端，仿佛就是为了拟合数据集中的点而生的，
对测试集中的数据表现就非常差。为了控制系数的size，让表达式看起来更像是阳间的求解结果，我们可以引入正则化的方法。
$$\omega^{*}=\arg\min \sum_{i=1}\limits^{n}(y_i-w^Tx_i)^2+\lambda\sum_{j=1}\limits^{d}w_j^2$$
这实际上就是拉格朗日算子，则我们跟之前一样可以将其写成矩阵形式：
$$(\boldsymbol y-X^T\omega)^T(\boldsymbol y-X^T\omega)+\lambda\omega^T\omega$$
对其求梯度可以得到
$$\nabla J_n(\alpha)=-2X(\boldsymbol y-X^T\alpha)+2\lambda\omega=0$$
则其最终的解就是： $$\omega^{*}=(XX^T+\lambda I)^{-1}Xy$$
这就是岭回归(Ridge
Regression)的方法，其中参数$\lambda$是可以自己确定的，我们可以保证矩阵$XX^T+\lambda I$是满秩的矩阵。

贝叶斯线性回归
--------------

现在我们重新考虑实际样本中可能会出现的噪声，即$y=f(\boldsymbol x,\omega)+\epsilon$，其中$\epsilon$服从$N(0,\sigma^2)$的正态分布。
根据贝叶斯定理可以得到：
$$P(\omega|y,x,\sigma)=\frac{P(y|\omega,x,\sigma)P(\omega|x,\sigma)}{P(y|x,\sigma)}$$
即posterior正比于prior和likelihood的乘积，即$\ln(posterior)\propto \ln(likelihood)\times\ln(prior)$，而在线性回归问题中，我们已经
知道了likelihood就是
$$l(D,\omega,\sigma)=-\frac{1}{2\sigma^2}\sum_{i=1}\limits^{n}(y_i-f(x_i,\omega)^2)+c(\sigma)$$
我们可以用如下方法选择： $$\begin{aligned}
    p(\omega) =N(w|0,\lambda^{-1}I)&=\frac{1}{(2\pi)^{\frac{d}{2}}|\lambda^{-1}I|^{\frac{1}{2}}}\exp^{-\frac{1}{2}\omega^T(\lambda^{-1}I)\omega}\\
    \ln(p(\omega)) &=-\frac{\lambda}{2}\omega^T\omega+c\end{aligned}$$
因此在贝叶斯理论下的岭回归模型需要优化的目标可以等价于：
$$-\frac{1}{2\sigma^2}\sum_{i=1}\limits^{n}(y_i-f(x_i,\omega)^2)+c(\sigma)-\frac{\lambda}{2}\omega^T\omega+c$$

逻辑回归 Logistic Regression
----

### 基本概念

逻辑回归往往选择通过一个sigmod函数将样本映射到某个区间上，以此来估计样本属于某种类别的概率从而达到分类的目的。我们经常选用的sigmod函数是：
$$y=\sigma(z)=\frac{1}{1+e^{-z}}$$
比如对于二分类问题，可以将样本映射到区间(-1,1)上，此时如果计算结果为正数则说明样本属于正例，反之就是反例，可以表示为：
$$\begin{aligned}
    P(y_i=1|x_i,\omega)=\sigma(\omega^Tx_i)=\frac{1}{1+e^{-\omega^Tx_i}}\\
    P(y_i=-1|x_i,\omega)=1-\sigma(\omega^Tx_i)=\frac{1}{1+e^{\omega^Tx_i}}\end{aligned}$$
上面的两个式子也可以统一写为： $$\begin{aligned}
    P(y_i=1|x_i,\omega)=\sigma(y_i\omega^Tx_i)=\frac{1}{1+e^{-y_i\omega^Tx_i}}\end{aligned}$$

### 参数估计

我们可以用极大似然法来估计参数$\omega$，依然用D表示数据集，具体的过程如下所示：
$$\begin{aligned}
        P(D) & =\prod_{i\in I}\sigma(y_i\omega^Tx_i)\\
        l(P(D)) & =\sum_{i\in I}\ln(\sigma(y_i\omega^Tx_i))=-\sum_{i\in I}\ln(1+e^{y_i\omega^Tx_i})
    \end{aligned}$$
因此我们可以将逻辑回归的极大似然法参数估计的loss函数定义成：
$$E(\omega)=\sum_{i\in I} \ln(1+e^{-y_i\omega^Tx_i})$$
对于一个二分类问题，我们如果用0和1而不是+1和-1来代表两种分类，那么上面的表达式又可以写为：
$$\begin{aligned}
        E(\omega)&=\sum_{i\in I\cap y_i=1}\ln(1+e^{-\omega^Tx_i}) + \sum_{i\in I\cap y_i=0}\ln(1+e^{\omega^Tx_i})\\
        &=\sum_{i\in I\cap y_i=1}\ln (e^{\omega^Tx_i})(1+e^{\omega^Tx_i}) + \sum_{i\in I\cap y_i=0}\ln (1+e^{-\omega^Tx_i})\\
        &=\sum_{i\in I} \ln(1+e^{\omega^Tx_i})-\sum_{i\in I \cap y_i=1}e^{\omega^Tx_i} \\
        &=\sum_{i\in I} (-y_i\omega^Tx_i+\ln(1+e^{\omega^Tx_i}))
    \end{aligned}$$
我们可以证明$E(\omega)$是一个关于$w$的凸函数，根据凸函数的可加性，我们只需要证明$-y_i\omega^Tx_i+\ln(1+e^{\omega^Tx_i})$是关于$w$的
凸函数，我们令$g(\omega)=-y_i\omega^Tx_i+\ln(1+e^{\omega^Tx_i})$则对其求一阶梯度可以得到：
$$\frac{\partial g(\omega)}{\partial\omega} = -y_ix_i+\frac{x_ie^{\omega^Tx_i}}{1+e^{\omega^Tx_i}}$$
能得到这个结果是因为我们有这样一个结论：

::: {.theorem}
**Theorem 1**.
*对于一个n维的向量$\omega$,我们有$\frac{\partial\omega^T}{\partial\omega}=I_{n}$*
:::

进一步地，我们对上面求的一阶梯度再求二阶梯度，可以得到：
$$\frac{\partial^2g(\omega)}{\partial\omega^2}=\frac{x_i^2e^{w^Tx_i}}{(1+e^{w^Tx_i})^2}\geq 0$$
因此我们证明了损失函数是一个凸函数，因此可以用梯度下降的方法求得其最优解，即：
$$\omega^{*}=\arg\min_{\omega} E(\omega)$$
根据上面求得的一阶梯度，可以得到基于梯度下降法的逻辑回归参数求解迭代方程：
$$\begin{aligned}
        \omega_{i+1}&=\omega_{i}-\eta(i)\sum_{i\in I}(-y_ix_i+\frac{x_ie^{\omega_i^Tx_i}}{1+e^{\omega_i^Tx_i}})\\
        &=\omega_i+\eta(i)\sum x_i(\frac{1}{1+e^{-\omega_i^Tx_i}}-y_i)\\
        &=\omega_i+\eta(i)X(\sigma(\omega_i, X)-y)
    \end{aligned}$$
其中$\sigma(x)$是sigmod函数，而$\eta(i)$是自己选择的学习率，一半是一个比较小的数字，并且应该不断减小。

感知机Perceptron
================

感知机模型的基本概念
--------------------

### 定义

感知机模型是一种二分类的线性分类模型，输入k维线性空间中的向量，输出一个实例的类别(正反类别分别用+1和-1来表示)，可以将这个分类过程
用一个函数来表示： $$y=f(x)=\mathrm{sign}(\omega x + b)$$
这里的参数$\omega$和$b$就是感知机的模型参数，其中$\omega$，而实数b是一个偏置(bias)，这里的sign函数是一个特殊的函数，当输入的
x是正数或者0的时候函数值就是1，输入的x是负数的时候函数值就是-1.

### 基于超平面的理解

可以把样本集里面的N个k维的向量看成是k维线性空间中的点，感知机的目标就是找到一个划分这个点集的超平面S，使得平面S的两侧分别是两种
类型的点，后面的测试集的分类就基于测试集中数据所对应的点和超平面的位置关系来划分，在正例的一侧$\omega x + b\ge 0$，反例的一侧
则小于0.因此感知机的学习目标就是根据样本数据学习出超平面的方程$\omega x + b=0$

### 基于神经元的理解

其实感知机可以看成是一种非常简单的二层神经网络，输入层的内容是k维向量的k个特征，输出层的结果就是向量的分类情况(分为+1和-1)两种，
而输出层的神经元存在一个阈值$\theta$，如果超过了这个阈值，神经元就会被激活，神经元存在一个激活函数$f(x)$，因此这个二层神经网络
可以用下面的式子来表示：
$$y=f(\sum_{i=1}^{k}\omega_ix_i-\theta)=f(\omega x-\theta)$$
其中$w_i$就是每个神经元的权重，也对应于定义式中的$\omega$，而阈值则是定义中的bias的相反数，函数f被称为激活函数，可以选用sign函数，
也可以选用sigmod函数，这两个函数的区别是sign是实数域$\mathbb R$不连续的函数，而sigmod是连续的，当选取sign函数作为激活函数的时候，
这个感知机模型的表达式就和超平面的理解中完全一致。

感知机模型的求解
----------------

### 损失函数

感知机的训练集必须要求是线性可分的，而感知机的的性能评估要考虑被误分类的点的情况，为了选择一个关于参数$\omega$和b连续的损失函数，
一个比较自然的选择就是根据各个点到超平面S的距离之和，即有：
$$L=-\frac{1}{||\omega||}\sum_{x_i\in M}y_i(\omega x_i+b)$$
这里的M表示数据集D中被误分类的点的集合，因此感知机中的损失函数可以定义为：
$$L(\omega, b)=-\sum_{x_i\in M}y_i(\omega x_i+b)$$
因此感知机模型的求解目标就变成了：
$$\min L(\omega,b)=-\sum_{x_i\in M}y_i(\omega x_i+b)$$

### 基于随机梯度下降法的求解

对损失函数求梯度可以得到：
$$\nabla_{\omega}L(\omega, b)=-\sum_{x_i\in M}y_ix_i$$
$$\nabla_{b}L(\omega, b)=-\sum_{x_i\in M}y_i$$
因此可以在学习的过程中，**先确定一组参数的初值，并选择好学习率$eta$**，(注意学习率不能超过1)，然后进行如下步骤开始学习：

-   在训练集中选取一组数据$(x_i,y_i)$

-   如果这组数据是误分类的，也就是说$y_i(\omega_kx_i+b_k)\le 0$，那么就要：
    $$w_{k+1}=w_k+\eta y_ix_i$$ $$b_{k+1}=b_k+\eta y_i$$

-   回到第二步继续循环，直到训练集中没有误分类的点，结束模型的训练

感知机的适用范围
----------------

支持向量机SVM和Kernel
=====================

支持向量的引入
--------------

对于一个分类问题，我们如果可以找到一个向量$\omega$，使得对于数据集D中的任何样本x，如果x是正例(用+1表示)就有$\omega^Tx> 0$，如果x是反例(用-1表示)
就有$\omega^Tx<0$，那么我们就可以很好地对数据集进行分类，判断依据就是$\omega^Tx$

我们也可以这样来考虑，将数据集D中的每个样本x投射到d维的空间上，如果我们可以找到一个d维空间里的超平面(hyperplane)$\omega^Tx+b=0$
将这个空间划分成两个部分，其中一个部分里的样本x全是正例，另一个空间里的样本x全是反例，那么这个数据集的分类问题就解决了，但是实际情况
并不会这么好，对于d维空间里的点x，其到超平面的距离可以表示为：
$$r=\frac{|\omega^Tx+b|}{||\omega||}$$
又为了能正确地进行分类，对于训练集中的数据，需要有： $$\begin{aligned}
    \left\{\begin{aligned}
        \omega^Tx_i+b \le -1, y_i=-1 \\
        \omega^Tx_i+b \ge +1, y_i=+1 \\
        \end{aligned}\right.\end{aligned}$$
并且存在一些样本使得上面的等号成立，我们就称这些使得等号取到的点称为支持向量(support
machine)，每个支持向量到超平面的距离是： $$r=\frac{1}{||\omega||}$$
则超平面两侧的两个支持向量到超平面的距离之和就是：
$$\gamma=\frac{2}{||\omega||}$$
这个量也被称为间隔(margin)，为了使得分类效果尽可能地好，我们应该让这个间隔尽可能地大，因此我们的目标可以转化为求$||\omega||^2$的
最小值，即 $$\begin{aligned}
        & \min_{\omega,b}\frac{1}{2}||\omega||^2\\
        & s.t. y_i(\omega^Tx_i+b)\ge 1
    \end{aligned}$$ 这个优化问题实际上就是支持向量机的基本形式

松弛变量slack variable
----------------------

在SVM问题求解中我们一般选择整个数据集中最小的一组间隔作为整个数据集的间隔，而我们优化的目标就是让这个最小间隔最大化。
但是现实往往没有这么美好，数据的分布不会像我们预想的那么完美，因此我们可以引入松弛变量(slack
variable)，给间隔一个 上下浮动的空间，即可以将问题转化成：
$$\begin{aligned}
        & \min\limits_{\omega,b} \frac{1}{2}||\omega||^2+C\sum\limits_{i=1}^n\xi_i\\
        & y_i(\omega^Tx_i+b)\ge 1-\xi_i\cap \xi_i\ge 0
    \end{aligned}$$ 我们可以将上面的约束条件转化为： $$\begin{aligned}
    \xi_i\ge 1- y_i(\omega^Tx_i+b)\cap \xi_i\ge 0\\
    \Rightarrow \xi_i = \max \lbrace 1- y_i(\omega^Tx_i+b),0\rbrace\end{aligned}$$
则优化的目标可以等价于：
$$\min\limits_{\omega,b}(\sum\limits_{i=0}^{n}\max(1-y(\omega^Tx_i+b),0)+\frac{1}{2C}||\omega||^2)$$
在求解SVM的过程中，可以将这个式子的前半部分作为loss
function，后半部分作为regularizer

线性模型的统一性
----------------

我们可以把上面的损失函数记作： $$l(f)=\max [1-yf, 0]$$
我们称这种类型的损失函数为hinge
loss(铰链损失函数)，因为其函数图像是一个类似于折线的形状。
则我们的优化目标可以写成：
$$\min\lbrace\sum_{i=0}\limits^{n}l(f)+\lambda R(f)\rbrace$$
其中R(f)是正则项。

事实上前面的所有线性模型，包括线性回归和逻辑回归的优化目标都可以写成上面的形式，区别在于loss函数的选择不同，
线性回归选择的loss是Square loss，而逻辑回归选择了Logistic
loss，这几种loss函数的图像也各有特点：

-   0-1 loss只有0和1两种值，在优化目标化简的时候特别方便

-   Square loss的波动幅度比较大，更能反映出极端情况下的损失

-   Logistic loss的变动比较平缓但是永远存在

-   Hinge loss在一定情况下会变成0，而在非0的时候比较平缓

核Kernel
--------

### 核方法 Kernel Method

支持向量机问题中，我们要求解的目标就是一个超平面$y=\omega^Tx+b$，用这个超平面来对数据集D进行线性的分割，这时候就出现了一个问题，
如果数据不能被线性分割该怎么办？事实上我们之前在SVM以及其他的线性模型中都默认了数据集D是线性可分的，如果实际情况中碰到的并非这么理想，
我们就应该采取一定的办法使得现有的数据变得线性可分。

核方法就可以解决这个问题，核方法通过将原始空间映射到一个更高维的特征空间中，使得数据集D中的样本x线性可分，而此时样本x也从一个d维向量映射
成了一个更高维度的向量(可以假定高维特征空间的维度是$\chi$)，用$\phi(x)$，则我们需要求解的问题变成了：
$$y=\omega^T\phi(x)+b$$

### 核函数 Kernel Function

而在使用核方法的时候经常会需要计算两个向量的内积，由于特征空间的维度可能很高甚至是无穷维，因此我们可以用一个求解简单的核函数来代替内积，
使得两个向量在特征空间的内积等于其原本的形式通过函数$K(x_i,x_j)$计算出来的结果，这就是kernel
trick

根据向量内积的性质我们可以推测出，和函数一定是对称的，并且运算的结果需要时非负数，即有：
$$K(x_i,x_j)=K(x_j,x_i)\ge 0$$
而对于$|D|=m$，$K(x_i,x_j)$可以形成m维的半正定矩阵，这个矩阵也被称为再生核希尔伯特空间(RKHS)，常见的核函数有：

-   线性核：$K(x_i,x_j)=x_i^Tx_j$

-   多项式核：$K(x_i,x_j)=(x_i^Tx_j)^d$

-   高斯核：$K(x_i,x_j)=\exp(-\frac{||x_i-x_j||^2}{2\sigma^2})$

-   拉普拉斯核：$K(x_i,x_j)=\exp(-\frac{||x_i-x_j||}{\sigma})$

-   Sigmod核：$K(x_i,x_j)=\tanh (\beta x_i^Tx_j+\theta)$

借助核方法和核函数我们可以把线性分类器用到非线性分类的问题上，但是具体如何选择核函数是未知的，或许需要自己一个个去尝试。

kNN：k-Nearest Neighbor分类
===========================

基本概念
--------

### kNN的基本思路

kNN：k-Nearest
Neighbor算法是一种分类算法，其核心想法是根据给定的训练集，对于需要判别的测试集中的样本，在训练集中找到与之最接近
的k个样本，并统计这k个样本的分类情况，将出现次数最多的类别作为测试集中的样本的分类结果。如果将k个最近的样本用$N_k(x)$来表示，那么
kNN的分类决策用公式表示就是：
$$y=\arg\max\sum_{x_i\in N_k(x)}I(y_i=c_i)$$
kNN的基本思路其实就是：如果一个东西看起来像鸭子，走路像鸭子，吃饭也像鸭子，那么它很可能就是一只鸭子。而\"最接近\"这个概念是有待商榷
的，因为我们没有统一的距离度量法则，因此需要确定一种度量距离的方法。此外k也需要自己选择，k=1的时候这个算法被称为最近邻算法

### 距离度量的选择

我们需要在n维的特征空间中度量两个点之间的距离，对于n维空间中的两个向量$x_i,x_j$，我们定义距离：
$$L_p(x_i,x_j)=(\sum_{l=1}^n|x_i^{(l)}-x_j^{(l)}|)^{\frac{1}{p}}$$
上面的公式中p=2的时候称为欧几里得距离，而p=1的时候称为曼哈顿距离，当p为无穷大的时候，L就是每个坐标中距离最大值.

### k的选择

k的选择也对最终的结果有比较大的影响，k选择太小的话学习时的误差会减小，但是估计误差会增大，预测结果会对样本的位置关系非常敏感。
如果k选的太大，则估计误差会降低，但是学习时的误差会增大，二者各有利弊，当然如果k=N那么就是按照出现次数最多的作为结果，完全
忽略了模型中的大量有用信息。

kd树：kNN求解
-------------

kd树是一种对k维空间中的实例点进行存储以便对其进行快速检索的树形数据结构，是一种二叉树，用来表示对k维空间的一个划分，使得每个
k维空间中数据集的点对应到一个k维空间超矩形上面去。

### kd树的构造

一般来说构造kd树的方法是先选择根节点，然后每次选定一个坐标轴，并用该方向的样本点坐标的中位数作为划分的依据，并沿着垂直于坐标
轴的方向作超矩形，然后将k维空间分出了新的两个区域，之后就在两个新的区域里面换一个坐标轴重复上面的操作，直到每个样本点都进行了
划分，每个子区域中没有样本点的时候停止。

用这种方法构造出来的kd树是

### kd树的搜索

对于一棵构造好的kd树进行k近邻搜索可以省去对大部分数据点的搜索，从而减少搜索的计算量，以最近邻为例，对于给定的目标点，搜索其最
邻近的点需要首先找到包含目标点的叶节点(注意，kd树里的节点代表的是超平面里的一块超矩形，不是样本点)，然后从该叶节点出发，回退到
父节点，并不断查找与目标点最邻近的节点，当确定不可能存在更近的节点的时候终止。

kNN实际编码体验
---------------

尝试了蔡老师机器学习课程中的kNN作业，目标是用kNN实现一个简单的验证码数字识别，主要任务其实就是用最原始的方法实现kNN，然后对数据集
进行一定的标注之后用kNN来识别数字，可能是因为场景比较简单，最后实现的验证准确率非常高。

决策树Decision Tree
===================

基本概念
--------

决策树(Decision
Tree)实际上是通过树形的结构来进行分类或者回归，在分类问题中可以根据样本的不同特征属性的不同属性值来进行多次
分类的决策最终确定输入样本的类别，也就相当于执行了一连串嵌套的if-else语句，**也可以被认为是定义在特征空间和类空间上的条件
概率分布**。决策树的优点是模型具有较好的可读性，分类的速度比较快。

决策树中有一系列节点和有向边，节点又分为内部节点和也节点，内部节点表示一个特征或者属性，叶节点表示一种类别。

### 决策树模型的建立

决策树主要依靠训练数据，采取损失函数最小化的原则来建立决策树模型，分为特征选择、决策树的生成和剪枝三个步骤。建立决策树的过程用
的特点主要有：

-   采用递归的方法构建决策树

-   当所有的样本都属于同一类或者在这一层已经对所有的特征都进行了分类，决策树的建立就停止

-   在决策树的每一层**选取一个最优的特征并用这个特征来进行分类**，这也是决策树算法最关键的地方

-   一般而言我们希望，随着划分过程的不断进行，决策树的分支节点所包含的样本尽可能属于同一类别，也就是节点的纯度越来越高

因为决策树考虑到了样本中的所有点，对所有的样本点都有正确的分类，因此决策树的bias实际上是0，因此评价一棵树的主要标准是variance，
一般来说规模比较小的决策树的performance更好。

最优特征的选取
--------------

上面已经说到决策树构建的最重要的部分就是在每一层选择合适的特征来进行划分，常见的算法有熵、信息增益等等。

### 信息熵Entropy

熵可以用来表示随机变量的**不确定性**。如果X是一个取值个数为有限个值即$P(X=x_i)=p_i$，那么随机变量X的熵的定义就是：
$$H(x)=-\sum_{i=1}^np_i \log p_i \in [0, \log n]$$
而对于随机变量X和Y，如果$P(X=x_i,Y=y_i)=p_{ij}$，那么在X给定的情况下随机变量Y的条件熵代表了X给定条件下Y的条件概率分布的
熵对于X的数学期望： $$H(Y|X)=\sum_{i=1}^np_iH(Y|X=x_i)$$
当用数据估计得到熵和条件熵的时候，他们又分别被称为经验熵和经验条件熵。

### 信息增益

信息增益(information
gain)表示**已经知道特征X的信息而使得类Y的信息的不确定度减少的程度**。特征A对于数据集D的信息增益$g(D,A)$定义为D的经验熵和D在A给定条件下的经验条件熵的差值，即：
$$g(D,A)=H(D)-H(D|A)$$ 信息增益又叫做**互信息(mutual
information)**，两个概念是等价的。基于信息增益的特征选择方法一般都是对于当前训练集D，计算其每个特征的信息增益，并比较大小，选择信息增益最大的特征来进行当前层的分类。

假设训练的数据集是$D$，有K个不同的分类$C_i$满足$\sum_{i=1}^K |C_k|=|D|$，其中特征A有n个不同的取值$a_i$，根据特征的取值将$D$划分成了n个不同的子集$D_i$，而每个子集$D_{ik}$中表示属于类别$C_k$的样本，那么数据集$D$的经验熵可以表示为：
$$H(D)=-\sum_{k=1}^{K}\frac{|C_k|}{|D|}\log_2\frac{|C_k|}{|D|}$$
特征A对于数据集D的条件经验熵可以表示为： $$H(D|A)=\sum_{i=1}^{n}\frac{|
D_i|}{|D|}H(D_i)=-\sum_{i=1}^{n}\frac{|
D_i|}{|D|}\sum^K_{k=1}\frac{|D_{ik}|}{|D_i|}\log_2\frac{|D_{ik}|}{|D_i|}$$

### 信息增益比

以信息增益作为数据集划分依据的时候，容易偏向于选择取值更多的特征作为分类标准，但是使用信息增益比，可以校正这一偏差，信息增益比是当前数据集D关于特征A的信息增益和自身的熵的比值，即：
$$g_R(D,A)=\frac{g(D,A)}{H_A(D)}=\frac{g(D,A)}{-\sum\limits_{i=1}^n\frac{|D_i|}{|D|}\log_2\frac{|D_i|}{|D|}}$$

### 基尼指数

基尼指数是另一种决策树的评价标准，对于分类问题，假设样本集合中有K中不同的类别，每类出现的概率为$p_k$那么基于概率分布的基尼指数
可以定义成 $$G(p)=\sum_{k=1}^Kp_k(1-p_k)=1-\sum_{k=1}^Kp_k^2$$
而在样本集合D中，基尼指数可以定义为：
$$G(D)=1-\sum_{k=1}^K\left(\frac{|C_k|}{|D|}\right)^2$$

决策树的剪枝与生成
------------------

### ID3算法

ID3算法的核心是在决策树的各个节点上用信息增益Information
Gain来进行特征的选择，并且递归地建立决策树：

-   从根结点开始，对于当前节点计算所有可能的特征值的信息赠一，选择信息增益最大的特征作为划分的一句并建立字节点

-   直到所有的特征的信息增益都很小或者没有子节点位置停止调用

### C4.5算法

和ID3算法类似，但是采用信息增益比作为选择的特征，其他的好像区别不是很大，但实际上决策树的代码实现应该是一系列统计学习的算法中难度比较大的，因为决策树需要使用树的数据结构，相比于其他算法的一系列矩阵计算在coding的过程中可能会有更大的麻烦。

### 决策树的剪枝Pruning

我们要明白决策树实际上是一种**基于大量统计样本的贪心算法**(仅根据个人理解，不代表任何工人观点)，在选定了一个决策标准(信息熵，信息增益，信息增益比等等)之后，就需要根据训练集来构建一棵决策树，构建的策略就是在每一次作出决策的时候(对应到树中就是产生了一系列子节点)都会依照给定的标准，计算每种特征相对于给定标准的值，然后选择当前区分度最大，也就是最优的特征作为当前层的决策依据。

但是这样一来，生成的决策树的复杂度可能是非常高的，因此我们应该想办法对这个决策树进行一定的简化，也就是进行剪枝，决策树的剪枝往往通过极小化决策树整体的损失函数或者代价函数来实现。设决策树T的每个叶节点t有$N_{kt}$个样本点，其中属于类别k的有个，节点t的经验熵可以表示为：
$$H_t(T)=-\sum_{k=1}^{n}\frac{N_{tk}}{N_t}\log \frac{N_{tk}}{N_t}$$
那么决策树的损失函数可以定义为：
$$C_{\alpha}(T)=\sum_{t=1}^{|T|}N_tH_t(T)+\alpha|T|=-\sum_{t=1}^{|T|}\sum_{k=1}^{n}N_{tk}\log \frac{N_{tk}}{N_t}+\alpha|T|$$
而其中$\alpha$是一个自己确定的参数值，比较大的$\alpha$可以使得损失函数偏向于更简单的模型，而比较小的则似的损失函数的优化偏向于比较复杂的模型，$\alpha=0$的时候完全无视了模型的复杂度，而只考虑模型和训练数据的匹配程度。其实这也就相当于给决策树的损失函数加一个正则项了，我们可以把过分精确的决策树看成是一种过拟合。

这里就不得不提一个冷知识：

::: {.tip}
****Tip** 1**.
*完整的决策树在训练的过程中分类的误差是0，也就是说不做任何剪枝处理的决策树对训练集的分类是一定准确的。*
:::

这也很好理解，因为决策树在构建的过程中并没有舍弃掉一些小概率的样本，而是对其如实进行了分类，虽然最后构建出来的决策树可能非常复杂但是在构建的时候因为我们的策略是贪心的，所以一定会把训练集上的样本分类到对应的叶节点上。

换句话说不剪枝的决策树的bias是0，而variance往往比较大，未剪枝的决策树在训练集上表现很好而在测试集上可能不尽人意，因此我们需要通过剪枝使得决策树可以更好地适应测试集，换句话说，**剪枝可以降低决策树模型的variance**

::: {.tip}
****Tip** 2**.
*剪枝过程是一个从底部向上的递归过程，首先需要计算每个节点的经验熵，如果剪掉某一段子树之后损失函数比剪掉之前更小，就进行剪枝，将父节点作为新的叶节点，重复下去直到不能继续为止。*
:::

CART算法
--------

### CART的定义

CART算法是Classification And Regression
Tree(分类与回归树)的简称，从字面意思就可以看出来这种决策树既可以用于分类，也可以用于回归。

-   CART树的一个基本假设就是决策树都是二叉树，**决策的结果只有是与否两种，这等价于递归地二分每个特征**，
    将输入空间划分成优先个但愿，并在这些单元上确定预测的概率分布，也就是在输入给定的条件下输出的条件概率分布

-   CART算法由生成和剪枝两个部分组成，对于生成的决策树希望它尽可能要大，剪枝的时候使用验证数据集来选择最优子树，这时候用损失函数最小作为剪枝的标准

### 回归树的生成

回归树对应的是**输入空间的一个划分和输出值的映射关系**，假如说输入的空间被划分成了M个单元$R_i$，每个分别对应了固定的输出值$c_i$，那么回归树模型可以表示为一个函数：
$$f(x)=\sum_{m=1}^{M}c_m\mathbb I(x\in R_m)$$
如果输入空间的划分确定，就可以用平方误差来估计训练数据的预测误差，用平方误差最小的准则来求解每个单元上的最优输出值。对于每一个单元$R_i$，这个单元对应的输出值就是其中若干个样本的label的平均值：
$$c_m=\frac{1}{|R_m|}\sum_{x_i\in R_m}y_i$$

因此问题在于如何对输入空间进行合理的划分，《统计学习方法》这本书里采用了一种启发式的方法，选择选择特征空间中的第j个变量$x^{(j)}$和它取的值s作为切分变量(splitting
variable)和切分点。并定义两个区域：
$$R_1(j,s)=\lbrace x|x^{(j)}\le s\rbrace \quad 
    R_2(j,s)=\lbrace x|x^{(j)}> s\rbrace$$
然后我们可以遍历所有的特征j和特征j的取值s来找到使得平房误差最小的切分点，寻找的依据是：
$$\min_{(j,s)}\left[\min_{c_1}\sum_{x_i\in R_1(j,s)}(y_i-c_1)^2+\min _{c_2}\sum_{x_i\in R_1(j,s)}(y_i-c_2)^2\right]$$

对于一个固定的j而言，上面的表达式中$c_1,c_2$的值是确定的，即按照上面所说的取这一范围内的平均值即可，而通过遍历所有的输入变量可以找到最优的切分变量j，然后就可以根据这样一个二元组将决策树划分成两个区域，然后在每个区域中重复上述操作，最后就可以生成一棵最小二乘回归树

### 分类树的生成

CART的分类树用基尼指数作为衡量集合不确定性的标准，因为CART默认采用的都是二分类，因此基尼指数可以定义成：
$$G(D,A)=\sum_{i=1}^2\frac{|D_i|}{|D|}G(D_i)$$
分类树的构建方法和普通的决策树一样，只不过选用基尼指数作为评价标准并且每次划分只能划分成两个子节点，递归建树的过程和普通决策树是完全一致的。

聚类Clustering
==============

聚类的基本概念
--------------

聚类是一种无监督学习，就是在给定的数据集中根据特征的相似度或者距离，将数据集中的对象归并到若干个类中(事先并不知道，即数据集是没有label的)，因此**解决聚类问题的关键是要先定义距离的度量方式**，因为很多时候数据点并没有物理意义上的"距离"(地图上的不同点拥有的是物理距离)，比如社交网络中人和人之间的关系就是一种广义上的距离，对于不同的场景，我们需要寻找不同的方式来定义数据样本中的"距离"

### 相似度和距离

用于聚类的样本数据集D或者样本集合，假设有n个样本，每个样本的有n个属性，那么样本的集合就可以用一个大小为$m\times n$的矩阵$X$来表示，矩阵的每一列表示一个样本的m个特征，常见的距离定义方法有

-   闵可夫斯基距离：对于两个向量，定义其闵可夫斯基距离为

$$d_{ij}=\left(\sum_{k=1}^m|x_{ik}-x_{jk}|^p\right)^{\frac 1p}$$

::: {.tip}
****Tip** 3**.
*闵可夫斯基距离中，当p=1时就是曼哈顿距离，p=2时就是欧式距离，p=$\infty$为时就是切比雪夫距离，等价于各个坐标数值差的绝对值*
:::

-   马哈拉诺比斯距离：假设样本矩阵X的协方差矩阵是S，则马哈拉诺比斯的距离共识可以表示为

$$d_{ij}=\left[(x_i-x_j)^TS^{-1}(x_i-x_j)\right]$$

::: {.tip}
****Tip** 4**.
*马哈拉诺比斯距离考虑了各个特征之间的相关性，通过协方差矩阵传递了这种相关性到两个样本的距离中，马哈拉诺比斯距离越大说明相似度越小*
:::

-   向量夹角的余弦：高中的时候立体几何中学的

$$s_{ij}=\frac{\left< x_i,x_j \right>}{||x_i||_2\times||x_j||_2}=\frac{\sum\limits_{k=1}^mx_{ki}x_{kj}}{\left(\sum\limits_{k=1}^mx_{ki}^2\sum\limits_{k=1}^mx_{kj}^2\right)^\frac 12}$$

-   相关系数：可以用相关系数来表示样本之间的相似度，(概率论和数理统计中似乎学过这个只是)其定义是：

$$r_{ij}=\frac{\sum\limits_{k=1}^m(x_{ki}-\bar x_i)(x_{kj}-\bar x_j)}{\sum\limits_{k=1}^m(x_{ki}-\bar x_i)^2\sum\limits_{k=1}^m(x_{kj}-\bar x_j)^2}$$

### 类和簇的判定

定义了什么是距离之后，我们遇到的第二个问题就是，对于一个数据集合G，当其中的点距离满足什么条件的时候可以将一系列数据样本定义成一个类(也可以叫簇，cluster)，而常见的定义方式有：

-   集合G中任意两个点的距离都不超过常数T

-   对于集合G中的任意一个点，都存在另一个样本使得二者的距离不超过常数T

-   对于G中的一个样本点，它和任意一个其他点的距离都不超过T

-   对于集合G中的样本，任意两个点之间的距离不超过V并且所有距离的平均值不超过T

我们可以用一些特征来刻画一个类，比如定义类的均值或者直径：
$$\bar x_G=\frac{1}{N_G}\sum_{i=1}^{N_G}x_i\qquad G_D=\max d_{ij}$$
同时类和类之间的距离也可以进行定义，这种距离又可以叫做连接(linkage)，可以有多种定义方式，比如最短距离，最长距离，中心局距离和平均距离等等。

层次聚类
--------

层次聚类是假设类别之间存在层次结构，将样本聚合到层次化的类里面去，分为自顶向下的聚类(分割)和自底向上的聚类(聚合)，聚合算法首先要确定三个要素，不同要素进行组合就可以得到一系列聚类算法：

-   定义距离或者相似度

-   确定合并的规则

-   确定算法停止的条件

### 聚合聚类

聚合聚类会根据输入的n个样本点进行聚类，最后输出一个包含所有样本的类和其中的各个子类，算法步骤如下：

-   对于n个样本点计算其距离，构成一个$m\times n$的矩阵

-   构造n个不同的类，每个类只包含一个样本

-   合并类间距离最小的两个类，以最短距离为类间距离，构造一个新类

-   计算新的类和当前各个类的距离，如果类的个数为1则停止计算，否则返回上一步

K-means算法
-----------

### 基本介绍

K-Means聚类(也叫做K均值聚类)是一种基于样本集合划分的聚类算法，k均值将样本集合划分称为k个子集，每个样本只属于一个类，因此是一种**硬聚类**的算法(相对的，软聚类算法是结果中每个样本可能属于多个类的算法)

对于给定的样本集合X，每个样本用特征空间中的特征向量来表示，我们希望将样本分成k个不同的类别$G_k$并且满足：
$$G_i\cap G_j=\emptyset\quad \bigcup\limits_{i=1}^{k}G_i=X$$
可以用C来表示这样的一个划分关系，而一个划分关系也就对应着一个K均值分类的结果，划分C是一个函数，可以表示为：
$$C(x_i)=G_k$$
表示一个样本的最终划分结果，而K均值算法在训练的过程中采用最小化损失函数的方式来选择最优的划分函数C

### K-Means的loss函数

K-Means采用欧几里德距离，并且用**所有点到其所在的类的中心的距离的平方和**作为损失函数：
$$W(C)=\sum_{l=1}^k\sum_{C(i)=l}||x_i-\bar x_l||^2$$
因此实际上K-Means算法的求解就是一个最优化的问题：
$$C^*=\arg\min W(C)=\arg\min\sum_{l=1}^k\sum_{C(i)=l}||x_i-\bar x_l||^2$$

### K-Means算法的训练过程

根据上面所确定的损失函数，K-Means算法的训练过程可以这样描述：

1.  首先要随机选K个样本作为初始的K个类的中心点，然后进行如下步骤

2.  对于每个样本点，计算其与K个类的中心点，然后在**选择最近的中心点所在的类作为该样本点的类**

3.  第2步的循环完成之后，在每个类中找到使得到该类中的所有样本点距离最小的中心点，使得该类中距离，也就是该类的样本中所有点的均值

4.  然后回到第2步用新的K个中心点计算出新的分类，重复上述步骤直到划分结果不再改变，就得到了K均值聚类的最终结果

### K-Means算法的特点

1.  K均值聚类属于启发式的方法，不能保证收敛到全局的最优，初始中心的选择会直接影响聚类的结果，要注意的时候类中心的移动往往不会有非常大的波动

2.  不同的类别数K会带来不同的聚类结果，一般来说分类的平均直径会随着K的增大先减小后不变，这个算法中的K是一个超参数，也是需要人为进行设定的

### k-Means代码实现

提供一份浙江大学蔡登老师给图灵班开设的ML课的作业中的一段K-means具体实现的代码，是本人写的因此可能比较垃圾

``` {.python language="python"}
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
```

提升Boosting与Ensemble
======================

基本介绍
--------

提升方法其实就是用多个不同的决策方法组合而成，根据不同方法的决策情况来进行最后的决策，也就是说不在一棵树上直接吊死，而是多找几棵树看看那棵树的风水最好再吊上去。

### 强可学习与弱可学习

如果可以用一个多项式复杂度的学习算法学习一个概念，并且正确率很高，那么这个概念就是强可学习的，如果学习的效果仅仅比随机猜测稍微好一点，那就是弱可学习的。如果预测的正确率还不如随机猜测，那这种算法属实没有存在的必要

弱可学习的算法一般比较容易学习，相比于强可学习算法更容易获得并且训练所需要的cost往往也更少，因此我们就产生了一个想法，是不是可以通过多个比较弱的学习算法一起预测结果，然后根据这些算法的预测结果再决定最后的分类结果。

-   但是决策的结果往往是少数服从多数，即"极大似然法"，我们暂且不讨论是否会出现多数人的暴政

-   可以给不同的分类器设定一个不同的权重，分类效果比较好的可以设置比较大的权重，这类通过组合多个算法共同预测的机器学习算法就是ensemble
    method

### ensemble有效性的证明

这种看似缝合怪的机器学习方法也叫做Combining Models，
顾名思义也就是将多个不同的model组合到一起共同预测结果，如果M个模型的权重是相同的，那么预测结果可以表示称：
$$y_{COM}=\frac 1 M \sum_{i=1}^My_i(x)$$ 那么为什么ensemble
method的集体决策效果要比单个分类器的预测效果好呢？我们可以假设真实的模型是$h(x)$，那么M个不同的模型组合而成的新模型的误差期望是：\
$$\begin{aligned}
        E_{COM}=E((y_{COM}-h(x))^2)=E((\frac 1 M \sum_{i=1}^My_i(x)-h(x))^2)\\=\frac 1{M^2}\sum_{i=1}^ME((y_i-h(x)^2)=\frac 1M E_{AV}
    \end{aligned}$$
我们发现通过ensemble方法，模型的variance变成了原来的$\frac 1M$，因此这种方法确实可以提高预测的准确性，因为它减小了模型的variance，使得模型的预测结果更加稳定。

如何生成多个模型：Bagging
-

我们现在已经知道模型的组合可以提高预测结果的稳定性，那么我们如何来生成多个模型进行组合呢？常见的方法有两种，一种是Bagging，另一种是Boosting

Bagging是通过不同的数据集划分方式来训练出多个模型，是一种可以并行的模型训练方式，它的两个关键步骤可以概括为：

-   Bootstrap
    sampling：从数据集D中生成若干组不同的大小为N的训练集，并训练出一系列基学习器(Base
    Learner)

-   Aggregating：即模型组合的策略，往往在分类模型中采用voting策略，即选择可能性最高的，而在回归问题中使用averaging，取平均数

Bagging方法在基学习器是弱可学习的时候提升效果非常好，这一点也可以理解，因为Bagging方法的本质是生成了一系列乌合之众，然后用voting策略来预测结果，但是乌合之众的规模变大的时候准确度也会有所提高。

这里的基学习器可以选择各类线性模型或者神经网络，也可以使用决策树，因为**决策树是一种比较容易获得的非线性分类器**，并且bias非常小而variance非常高，我们用ensemble的方式刚好可以降低决策树的variance，得到bias和variance都比较小的模型，而这种使用决策树作为基学习器的Bagging模型叫**做随机森林(Random
Forest)**，这个名字也比较形象，很多"树"聚集成了森林。

Boosting方法与AdaBoost
----------------------

与Bagging方法不同Boosting方法是一种迭代式的训练算法，其中比较著名的就是AdaBoost，这是一种串行的Ensemble方法，通过若干次迭代的训练将训练数据学习成若干个弱分类器，然后通过调整权重来提高模型预测的准确度。

### AdaBoost算法步骤

1.  将训练集D中的N个训练数据赋予初始的权重，一般来说是所有样本之间都相等，即
    $$D_1=(w_{11},w_{12},\dots,w_{1N})\quad w_{1i}=\frac 1N$$

2.  对数据进行一共M轮的迭代，第m轮的时候训练数据的权重分布是$D_m$，并训练得到一个弱分类器$G_m(x)$，然后计算这个分类器的训练误差：
    $$e_m=\sum_{i=1}^NP(G_m(x_i)\not=y_i)=\sum_{i=1}^Nw_{mi}\mathbb I(G_m(x_i)\not=y_i)$$

3.  然后计算该分类器的系数：
    $$\alpha_m=\frac 12\log \frac {1-e_m}{e_m}$$

4.  更新整个训练集的权重分布$D_{m+1}=(w_{m+1,1},w_{m+1,2},\dots,w_{m+1,N})$，其中$Z_m$叫做规范化因子，很明显这样的规范化因子使得前面的系数的和变成了1，也就是说权重系数成为了一个概率分布
    $$\begin{aligned}
                w_{m+1,i}=\frac{\exp(-\alpha_my_iG_m(x_i))}{Z_m}w_{m,i}\\Z_m=\sum_{i=1}^N\exp(-\alpha_my_iG_m(x_i))w_{m,i}
            \end{aligned}$$

5.  最后生成一个各个若分类器进行线性组合，形成权重不同的强分类器：
    $$\begin{aligned}
                f(x)&=\sum_{i=1}^M\alpha_mG_m(x)\\G(x)&=\mathrm{sign}(f(x))=\mathrm{sign}(\sum_{i=1}^M\alpha_mG_m(x))
            \end{aligned}$$

### AdaBoost误差分析

AdaBoost存在一个误差上界：
$$\frac 1N\sum_{i=1}^N\mathbb I(G_m(x_i)\not=y_i)\le \frac 1N\sum_i\exp(-y_if(x_i))=\prod_mZ_m$$
而对于一个二分类的问题，有：\
$$\begin{aligned}
        \prod_{m=1}^MZ_m&=\prod_{m=1}^M2\sqrt{e_m(1-e_m)}\\ 
        &=\prod_{m=1}^M\sqrt{1-4\gamma_m^2}\le\exp(-2\prod_{m=1}^M\gamma^2_m)
    \end{aligned}$$ 其中$\gamma_m=\frac 12-e_m$

EM算法和高斯混合模型GMM
=======================

EM算法的全称是Expectation
Maximization，是用于**对含有隐变量的概率模型进行参数极大似然估计的一种迭代算法**，隐变量就是数据集的特征中中没给出但是会影响预测结果的变量。如果没有隐变量，那么在概率模型的估计中直接进行极大似然估计，然后求导即可，而有隐变量存在的时候导数不能直接求出，所以才需要使用EM算法。EM算法主要分成E和M两个主要步骤：

-   E步骤：求解参数估计值的期望

-   M步骤：使用极大似然法求出期望的最值，然后将得到的结果放到下一轮迭代中去

EM算法的推导
------------

如果说一个概率模型中中含有可观测隐变量Z，那么我们在极大化观测数据Y关于模型参数的似然函数的时候，实际上的目标就变成了
$$\begin{aligned}
        L(\theta)&=\log P(Y|\theta)=\log \sum_{Z}P(Y,Z|\theta)\\ &=\log \sum_{Z}P(Y|Z,\theta)P(Z|\theta)
    \end{aligned}$$
而隐变量是没有观察到的，因此不能直接使用极大似然法进行参数估计，EM算法要解决的就是在隐变量条件下的参数估计问题，两个步骤的具体过程如下：

### EM算法的输入输出要求

一般来说用Y表示观测随机变量的数据，用Z来表示隐变量的数据，Y和Z联合在一起一般称为完全数据，观测数据Y又叫做不完全数据，EM算法的求解需要知道观测变量数据Y，隐变量数据Z，联合概率分布$P(Y,Z|\theta)$，条件分布$P(Z|Y,\theta)$，最终输出的是模型的参数$\theta$

### Q函数

完全数据的对树似然函数$\log P(Y,Z|\theta)$关于在给定观测数据Y和当前参数估计$\theta_i$下对于未观测数据Z的条件概率分布$P(Z|Y,\theta)$的期望称为Q函数，也就是：
$$Q(\theta,\theta_i)=E_Z[\log P(Y,Z|\theta)|Y,\theta_i]$$

### EM算法求解过程

在选定好初始化参数$\theta_0$之后，需要进行：

1.  E步骤：假设第i次迭代的时候参数为$\theta_i$，则在第i+1次的E步骤，计算Q函数：
    $$\begin{aligned}
                Q(\theta,\theta_i)&=E_Z[\log P(Y,Z|\theta)|Y,\theta_i]\\
                &=\sum_{Z}\log P(Y,Z|\theta)P(Z|Y,\theta_i)
            \end{aligned}$$

2.  M步骤：求使得$Q(\theta,\theta_i)$最大化的$\theta$作为本次迭代的新估计值：
    $$\theta_{i+1}=\arg\max Q(\theta,\theta_i)$$

3.  重复E步骤和M步骤直到得到的参数收敛

### EM算法的有效性和收敛性证明

这部分内容《统计学习方法》上有比较详细的证明，奈何暂时看不懂，就先不管了，先理解EM算法的基本步骤再说。

高斯混合模型GMM
---------------

高斯混合模型是指如下形式的概率分布模型：
$$P(y|\theta)=\sum_{k=1}^K\alpha_k\phi(y|\theta_k)$$
其中$\alpha_i$表示权重系数，$\phi(y|\theta_k),\theta_k=(\mu_k,\sigma^2_k)$表示一个高斯分布的密度函数，即：
$$\phi(y|\theta_k)=\frac{1}{\sqrt{2\pi }\sigma_k}\exp\left(-\frac{(y_k-\mu_k)^2}{2\sigma^2_k}\right)$$

### GMM的隐变量分析

高斯混合模型中的模型参数有:$\theta=(\alpha_a,\dotsm\alpha_K,\theta_1,\dots,\theta_K)$，我们需要估计每个高斯分量的权重和自身的模型参数，而GMM中是有隐变量的，我们可以这样来分析：首先根据每个不同的权重$\alpha_k$来选择第k个分量计算生成的观测数据$y_j$，这个时候的观测结果是已知的，但是反应观测数据来自第k个分量的数据是未知的，因此我们可以用一个隐变量$\gamma_{jk}$来表示第j个观测来自于第k的模型，因此该隐变量只能取0和1两个值。

这里很重要的一点，也是非常容易出现的误区就是，观测数据的生成并不是在K个高斯模型中分别生成然后按照权重比例组合，而是以权重比例为概率分布情况，随机选择出一个高斯模型来生成观测数据$y_j$，因此隐变量$\gamma_{jk}(k=1,2,\dots,k)$中有且仅有一个是1，剩下的都是0

### GMM的求解和参数估计

EM求解GMM需要先输入N个观测数据，并输出一个高斯混合模型的参数，训练的过程主要分为E步骤和M步骤：

-   E步骤：依据当前的模型参数，计算分模型k对观测数据的响应度，即：

$$\hat \gamma_{jk}=\frac{\alpha_k\phi(y_j|\theta_k)}{\sum\limits_{k=1}^K\alpha_k\phi(y_j|\theta_k)}$$

-   M步骤：根据隐变量计算出新一轮迭代所需要的模型参数，模型就根据这些参数产生：

$$\mu_k=\frac{\sum\limits_{j=1}^N\hat\gamma_{jk}y_j}{\sum\limits_{j=1}^N\gamma_{jk}}$$
$$\sigma_k^2=\frac{\sum\limits_{j=1}^N\hat \gamma_{jk}(y_j-\mu_k)^2}{\sum\limits_{j=1}^N\gamma_{jk}}$$
$$\alpha_k=\frac{\sum\limits_{j=1}^N\hat \gamma_{jk}}{N}$$
其实这一部分一开始我没怎么看懂，后面有空继续看。

主成分分析PCA
=============

基本概念
--------

主成分分析(Principal component analysis,
PCA)是一种常见的无监督学习方式，**利用正交变换把线性相关变量表示的观测数据转换成几个线性无关变量所表示的数据**

-   这些线性无关的变量叫做主成分

-   主成分分析属于降维方法的一种，关于这个，统计学习方法中举了很多例子

在数据总体上进行的主成分分析叫做总体主成分分析，在有限的样本数据上进行的主成分分析称为样本主成分分析

样本主成分分析
--------------

问题的情景：对m维的随机变量x进行n次独立的观测，得到一个大小为$m\times n$的样本矩阵X，并且可以估计这些样本的均值向量：
$$\bar x=\frac 1n \sum_{i=1}^nx_i$$

### 样本的统计量

对于上面的样本，协方差矩阵可以表示为：
$$s_{ij}=\frac 1{n-1}\sum_{k=1}^n(x_{ik}-\bar x_i)(x_{jk}-\bar x_j)$$
而样本的相关矩阵可以写成$R=[r_{ij}]_{m\times m}$：
$$r_{ij}=\frac{s_{ij}}{\sqrt {s_{ii}s_{jj}}}$$

### 主成分的定义

我们可以设一个m维的随机变量x到m维的随机变量y的一个线性变换：
$$\mathcal Y=(y_1,y_2,\dots,y_m)=A^Tx=\sum_{i=1}^m\alpha_ix_i$$
$$y_i=\alpha_i^Tx=\sum_{j=1}^m\alpha_{ji}x_j$$

### 主成分的统计量

对于随机变量$\mathcal Y=(y_1,y_2,\dots,y_m)$，其统计量有：
$$\begin{aligned}
        \bar y_i=\frac 1n\sum_{j=1}^n\alpha_i^Tx_j=\alpha_i \bar x \\ \mathrm{var}(y_i)=\alpha_i^T S\alpha_i \\ \mathrm{cov}(y_i,y_j)=\alpha_i^T S\alpha_j
    \end{aligned}$$

### PCA的具体算法步骤

-   先将样本矩阵进行normalization：
    $$x_{ij}^*=\frac{x_{ij}-\bar x_i}{\sqrt{s_{ij}}}$$

-   计算样本的协方差矩阵$XX^T$并对其进行特征值的分解

-   取最大的d个特征值所对应的特征向量$w_j$

-   输出投影矩阵$W^*=(w_1,w_2,\dots,w_d)$

### PCA代码实现

``` {.python language="python"}
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
```

线性判别分析LDA
===============

线性判别分析(Linear Disciminant
Analysis)是一种经典的降维方法，并且是一种线性的学习方法，LDA的基本想法就是，对于给定的训练集，我们可以设法将这些样本投影到一条直线上，并且使得同类样本点的投影尽可能接近而不同类的样本点的投影尽可能远离，这样一来我们实际上就需要训练出一条直线来进行特征空间中的分类，而对于测试集中的数据，可以将其同样投影到这条直线上面去，再根据投影点位置来确定该样本属于哪一类别。

LDA问题的定义
-------------

我们先来研究二分类问题的LDA，可以假设问题定义在数据机D上，用$X_i,\mu_i,\Sigma_i$分别表示某一类别的数据集合，均值和协方差矩阵，如果能够将数据投影到特征空间中的一条直线$\omega$上，那么两类样本的中心在直线上的投影分别是$\omega ^T \mu_0$和$\omega ^T \mu_1$，如果将两类样本的所有点都投影到这条直线上面，那么两类样本的协方差是$\omega ^T\Sigma_i\omega$

而我们的目的是希望同类别的点在直线上的投影尽可能靠近而不同类的尽可能远离。因此可以让同类别的投影点的协方差尽可能小，即一个目标可以定义成：
$$\min \left(\omega ^T\Sigma_0\omega+\omega ^T\Sigma_1\omega\right)$$
而同时我们也希望不同类别的数据样本点的投影尽可能远离，因此可以让类中心之间的距离尽可能大，则另一个目标可以表示为：
$$\max ||\omega ^T \mu_0-\omega ^T \mu_1||^2_2$$
这样一来，我们可以结合两个优化目标，定义目标函数：\
$$\begin{aligned}
        J &=\frac{\left\|{\omega}^{{T}} {\mu}_{0}-{\omega}^{{T}} {\mu}_{1}\right\|_{2}^{2}}{{\omega}^{{T}}\left({\Sigma}_{0}+{\Sigma}_{1}\right) {\omega}} 
        =\frac{\left\|\left({\omega}^{{T}} {\mu}_{0}-{\omega}^{{T}} {\mu}_{1}\right)^{{T}}\right\|_{2}^{2}}{{\omega}^{{T}}\left({\Sigma}_{0}+{\Sigma}_{1}\right) {\omega}} \\
        &=\frac{\left\|\left({\mu}_{0}-{\mu}_{1}\right)^{{T}} {\omega}\right\|_{2}^{2}}{{\omega}^{{T}}\left({\Sigma}_{0}+{\Sigma}_{1}\right) {\omega}} 
        =\frac{\left[\left({\mu}_{0}-{\mu}_{1}\right)^{{T}} {\omega}\right]^{{T}}\left({\mu}_{0}-{\mu}_{1}\right)^{{T}} {\omega}}{{\omega}^{{T}}\left({\Sigma}_{0}+{\Sigma}_{1}\right) {\omega}} \\
        &=\frac{{\omega}^{{T}}\left({\mu}_{0}-{\mu}_{1}\right)\left({\mu}_{0}-{\mu}_{1}\right)^{{T}} {\omega}}{{\omega}^{{T}}\left({\Sigma}_{0}+{\Sigma}_{1}\right) {\omega}}
    \end{aligned}$$

LDA的优化与求解
---------------

### 问题的改写

对于上面得到的优化目标，我们可以定义：

::: {.definition}
**Definition 3**. *类内散度矩阵(within-class scatter
matrix)，用来表示同一个类别内的点的接近程度*
:::

$$S_w=\Sigma_0+\Sigma_1=\sum_{x\in X_0}(x-\mu_0)(x-\mu_0)^T+\sum_{x\in X_1}(x-\mu_1)(x-\mu_1)^T$$

::: {.definition}
**Definition 4**. *类间散度矩阵(between-class scatter
matrix)，用来表示不同类别内的点的远离 程度*
:::

$$S_b=(\mu_0-\mu_1)(\mu_0-\mu_1)^T$$ 这样一来上面的目标函数就可以变成：
$$J=\frac{\omega^TS_b\omega}{\omega^TS_w\omega}$$
这就是一种广义瑞丽商的优化形式。

### 瑞丽商Rayleigh quotient

对于$n\times n$的矩阵A，瑞丽商的定义形式是：
$$R(A,x)=\frac{x^TAx}{x^Tx}$$
瑞丽商这种形式的函数很重要的一个性质就是其最大值等于A的最大的特征值，而最小值等于A最小的特征值，即：
$$\lambda_{\min}\le R(A,x)\le\lambda_{\max}$$
可以用拉格朗日乘子法来证明这一结论，我们不妨设$x^tx=1$，这样一来根据拉格朗日乘子法，优化的目标函数可以写成：
$$f(x)=x^TAx+\lambda(x^Tx-1)$$ 求梯度并令梯度等于0可以得到：
$$\frac{\partial{f(x)}}{\partial{x}}=Ax-\lambda x = 0 \rightarrow Ax=\lambda x$$
这其实就是矩阵特征值的定义形式，这里所有能够使得梯度为0的x就是A的所有特征值对应的特征向量，这样一来$x^TAx$在极值点的时候的计算结果就是A的一系列特征向量，因此该函数的最大值就是A最大的特征向量，而最小值就是A最小的特征向量。

而广义瑞丽商的定义是： $$R(A,B,x)=\frac{x^TAx}{x^TBx}$$
广义瑞丽商可以变形为：
$$R(A,B,x)=\frac{x^TAx}{x^TBx}=\frac{\hat x^T(B^{-\frac 12}AB^{-\frac 12})\hat x}{\hat x^T \hat x}$$
同上面一样可以得到这个函数的最大值和最小值分别是矩阵$B^{-\frac 12}AB^{-\frac 12}$的最大的特征值和最小值。

### LDA的求解

根据瑞丽商的性质，我们可以知道LDA的求解就是对矩阵$S_w^{-\frac 12}S_bS_w^{-\frac 12}$进行特征值分解，最大的特征值对应的就是最大值，最小的特征值对应的就是最小值，同时又可以进行进一步的变形：
$$\omega = S_w^{-1}(\mu_0-\mu_1)$$
同时LDA也可以用贝叶斯决策理论来解释，并且可以证明，当两类数据满足同先验、满足高斯分布并且协方差相等的时候，LDA可以达到最优的分类结果。

高斯分布下的LDA
---------------

### 问题的定义

这一节内容从从贝叶斯决策理论来推导高斯分布下的LDA模型，
如果样本满足高斯分布，我们先考虑二分类时候的情况，可以设两类样本分别满足高斯分布并且共享协方差矩阵，这样一类：
$$\begin{aligned}
        P(x|y=c,\theta)=\mathcal{N}(x|\mu_c,\Sigma)\quad c\in \left\{0, 1\right\}
    \end{aligned}$$ 这样一来，根据贝叶斯公式，后验概率分布可以表示为：
$$\begin{aligned}
        P(y=c|x,\theta)=&\frac{P(x|y=c,\theta)P(y=c,\theta)}{\sum _{c\in C}P(x|y=c,\theta)P(y=c,\theta)}\\
        \propto & \quad \pi_{c} \exp \left[{\mu}_{c}^{T} {\Sigma}^{-1} {x}-\frac{1}{2} {x}^{T} {\Sigma}^{-1} {x}-\frac{1}{2} {\mu}_{c}^{T} {\Sigma}^{-1} {\mu}_{c}\right] \\
        = & \exp \left[{\mu}_{c}^{T} {\Sigma}^{-1} {x}-\frac{1}{2} {\mu}_{c}^{T} {\Sigma}^{-1} {\mu}_{c}+\log \pi_{c}\right] \exp \left[-\frac{1}{2} {x}^{T} {\Sigma}^{-1} {x}\right]
    \end{aligned}$$ 这里我们可以令： $$\begin{array}{l}
        \gamma_{c}=-\frac{1}{2} {\mu}_{c}^{T} {\Sigma}^{-1} {\mu}_{c}+\log \pi_{c} \\
        {\beta}_{c}={\Sigma}^{-1} {\mu}_{c}
    \end{array}$$ 这样一来要求的后验概率就可以写成：
$$p(y=c \mid {x}, {\theta})=\frac{e^{{\beta}_{c}^{T} {x}+\gamma_{c}}}{\sum_{c^{\prime}} e^{{\beta}_{c^{\prime}}^{T} {x}+\gamma_{c^{\prime}}}}=\mathcal{S}({\eta})_{c}$$
这其实就是一个softmax函数的形式，softmax函数可以把一个分布标准化成一个和为1的概率分布，而**sigmoid函数其实就是一个一元形式的softmax函数**

### 二分类下的特殊情况

对于二分类问题，我们可以进行这样的变形： $$\begin{aligned}
        P(y=1 \mid {x}, {\theta}) &=\frac{e^{{\beta}_{1}^{T} {x}+\gamma_{1}}}{e^{{\beta}_{1}^{T} {x}+\gamma_{1}}+e^{{\beta}_{0}^{T} {x}+\gamma_{0}}} \\
        &=\frac{1}{1+e^{\left({\beta}_{0}-{\beta}_{1}\right)^{T} {x}+\left(\gamma_{0}-\gamma_{1}\right)}}=\operatorname{sigmoid}\left(\left({\beta}_{1}-{\beta}_{0}\right)^{T} {x}+\left(\gamma_{1}-\gamma_{0}\right)\right)
        \end{aligned}$$
这样就可以将二分类情况下的高斯分布转化成一个sigmoid函数的形式，根据之前的$\beta,\gamma$的定义，我们有：
$$\begin{aligned}
        \gamma_{1}-\gamma_{0} &=-\frac{1}{2} {\mu}_{1}^{T} {\Sigma}^{-1} {\mu}_{1}+\frac{1}{2} {\mu}_{0}^{T} {\Sigma}^{-1} {\mu}_{0}+\log \left(\pi_{1} / \pi_{0}\right) \\
        &=-\frac{1}{2}\left({\mu}_{1}-{\mu}_{0}\right)^{T} {\Sigma}^{-1}\left({\mu}_{1}+{\mu}_{0}\right)+\log \left(\pi_{1} / \pi_{0}\right)
    \end{aligned}$$ 因此我们可以定义： $$\begin{aligned}
        {\omega} &={\beta}_{1}-{\beta}_{0}={\Sigma}^{-1}\left({\mu}_{1}-{\mu}_{0}\right) \\
        {x}_{0} &=\frac{1}{2}\left({\mu}_{1}+{\mu}_{0}\right)-\left({\mu}_{1}-{\mu}_{0}\right) \frac{\log \left(\pi_{1} / \pi_{0}\right)}{\left({\mu}_{1}-{\mu}_{0}\right)^{T} {\Sigma}^{-1}\left({\mu}_{1}-{\mu}_{0}\right)}
    \end{aligned}$$ 这样一来我们就可以得到(这个比较好推出)：
$$\omega^Tx_0=-(\gamma_1-\gamma_0)$$ 因此，后验概率可以被写成：
$$P(y=1|x,\theta)=\mathrm{sigmoid}(\omega^T(x-x_0))$$
我们可以将$\omega^T(x-x_0)$看成是特征空间中的点$x$平移了$x_0$个单位之后投影到直线$\omega$上面，而sigmoid函数则是对投影的结果进行一个分类，看投影点是更靠近哪一类别。这个形式就是线性判别分析的形式，即将样本点投影到一条直线上，然后看距离多个类别的中心点的距离来判断样本的类别。

### 参数估计和模型优化

我们可以来考虑用极大似然法来训练一个模型，我们可以将该问题的对数似然函数定义为：
$$\log P(\mathcal{D} \mid \boldsymbol{\theta})=\left[\sum_{i=1}^{N} \sum_{c=1}^{C} \mathbb{I}\left(y_{i}=c\right) \log \pi_{c}\right]+\sum_{c=1}^{C}\left[\sum_{i: y_{i}=c} \log \mathcal{N}\left(x \mid \boldsymbol{\mu}_{c}, \boldsymbol{\Sigma}_{c}\right)\right]$$
这里的参数可以用样本的数据进行估计，比如$\pi_c=\frac{N_c}{N}$，$\mu_c=\frac{1}{N_c}\sum_{i\in N(c)}x_i$，$\Sigma_c=\frac 1N_c(x-\mu_c)(x-\mu_c)^T$，这些都是基于极大似然法的常见估计。
