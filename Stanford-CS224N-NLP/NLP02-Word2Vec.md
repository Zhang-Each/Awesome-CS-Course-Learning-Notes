NLP02-word2vec
=================



Word2vec
--------

### 基于迭代的模型

基于SVD的方法实际上考虑的都是单词的**全局特征**，因此计算代价和效果都不太好，因此我们可以尝试建立一个基于迭代的模型并根据单词在上下文中出现的概率来进行编码。

另一种思路是设计一个参数是词向量的模型，然后针对一个目标进行模型的训练，在每一次的迭代中计算错误率或者loss函数并据此来更新模型的参数，也就是词向量，最后就可以得到一系列结果作为单词的嵌入向量。

常见的方法包括**bag-of-words(CBOW)和skip-gram**，其中CBOW的目标是**根据上下文来预测中心的单词**，而**skip-gram则是根据中心单词来预测上下文中的各种单词出现的概率**。这两种方法统称为word2vec，是Google AI团队在[《Distributed Representations of Words and Phrases and their Compositionality》](https://papers.nips.cc/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf) 等多篇论文中提出的词向量训练模型。

而这些模型常用的训练方法有：

- 层级化(hierarchical)的softmax

-   负采样negative sampling：判断两个单词是不是一对上下文与目标词，如果是一对，则是正样本，如果不是一对，则是负样本，而采样得到一个上下文词和一个目标词，生成一个正样本，用与正样本相同的上下文词，再在字典中随机选择一个单词，这就是负采样



### 语言模型Language Model

**语言模型**可以给一系列单词序列指定一个概率来判断它是不是可以作为一个完整的句子输出，一般完整性强的，语义通顺的句子被赋予的概率会更大，而无法组成句子的一系列单词可能就会计算出非常小的概率，语言模型用公式来表示就是要计算下面的概率：
$$
P\left(w_{1}, w_{2}, \cdots, w_{n}\right)
$$
如果我们假设每个单词在每个位置出现的概率是**独立**的话，这个表达式就会变成：
$$
P\left(w_{1}, w_{2}, \cdots, w_{n}\right)=\prod_{i=1}^nP(w_{i})
$$
但这显然是不可能的，因为单词的出现并不是不受约束的，而是和其周围的单词(也就是上下文)有一定的联系，这也是NLP中一个非常重要的观点，那就是

> 分布式语义(Distributional semantics)：一个词语的**意思是由附近的单词决定的**，一个单词的上下文就是"附近的单词"，可以通过一个定长的滑动窗口来捕捉每个单词对应的上下文和语义。

Bi-gram model认为单词出现的概率取决于它的上一个单词，即：
$$
P\left(w_{1}, w_{2}, \cdots, w_{n}\right)=\prod_{i=2}^nP(w_{i}|w_{i-1})
$$
但显然这也是一种非常的的模型，说白了就是一个一阶的马尔科夫链，只是一种非常理想化特殊化的模型，因此我们需要寻找更合适的语言模型。

事实上上面提出的CBOW和Skip-gram也可以看成是一种简单的语言模型，我们通过训练一个语言模型的方式来训练出文档中出现的单词的词向量。

### 连续词袋模型(CBOW)

连续词袋模型(Continuous Bag of Words Model)是一种根据上下文单词来预测中心单词的模型，对于每一个单词我们需要训练出两个向量u和v分别代表该单词作为中心单词和作为上下文单词的时候的嵌入向量(一种代表输出结果，一种代表输入)

因此可以定义两个矩阵$\mathcal{V} \in \mathbb{R}^{n \times|V|}$ 和 $\mathcal{U} \in \mathbb{R}^{|V| \times n}$，其中n代表了嵌入空间的维度，矩阵$\mathcal{V}$的每一列代表了词汇表中一个单词**作为上下文单词时候的嵌入向量**，用$v_i$表示，而矩阵$\mathcal{U}$表示的是预测的结果，每一行代表了一个单词的嵌入向量，用$u_j$表示。

这里其实隐含了一个假设，那就是单词作为上下文和中心词的时候具有的特征是不一样的，因此要用不同的词向量来表示。

#### CBOW模型的工作原理

1.  首先根据输入的句子生成一系列one-hot向量，假设要预测的位置是c，上下文的宽度各为m，则生成的一系列向量可以表示为：
    $$\left(x^{(c-m)}, \ldots, x^{(c-1)}, x^{(c+1)}, \ldots, x^{(c+m)} \in \mathbb{R}^{|V|}\right)$$

2.  计算得到上下文单词的嵌入向量： $$v_j=\mathcal{V}x^{(j)}$$

3.  求出这些向量的均值：
    $$\hat{v}=\frac{v_{c-m}+v_{c-m+1}+\ldots+v_{c+m}}{2 m} \in \mathbb{R}^{n}$$

4.  生成一个**分数向量**$z=\mathcal{U}\hat{v}$，当相似向量的点积较大时，会将相似的词推到一起，以获得较高的分数

5.  用softmax函数将分数向量转化成概率分布：$\hat{y}=\mathrm{softmax}(z)\in \mathbb R^{|V|}$

6.  根据概率分布$\hat{y}$得到最终的结果$y$(是一个one-hot向量的形式)，也就是最有可能出现在中心位置的单词

因此现在的问题就变成了，我们应该如何学习到两个最关键的矩阵$\mathcal{V}$和$\mathcal{U}$，这也是CBOW最关键的一个问题，这里可以采用**交叉熵**来定义一个目标函数：
$$
H(\hat{y}, y)=-\sum_{j=1}^{|V|} y_{j} \log \left(\hat{y}_{j}\right)=-y_{i} \log \left(\hat{y}_{i}\right)
$$
因为概率的**真实值$y$是一个one-hot向量，只有一个维度是1其他的都是0**，因此可以进行这样的化简。



#### 损失函数与优化方式

我们可以定义如下形式的损失函数并进行优化：
$$
\begin{aligned}
	\min J &=-\log P\left(w_{c} \mid w_{c-m}, \ldots, w_{c-1}, w_{c+1}, \ldots, w_{c+m}\right) \\
	&=-\log P\left(u_{c} \mid \hat{v}\right) \\
	&=-\log \frac{\exp \left(u_{c}^{T} \hat{v}\right)}{\sum_{j=1}^{|V|} \exp \left(u_{j}^{T} \hat{v}\right)} \\
	&=-u_{c}^{T} \hat{v}+\log \sum_{j=1}^{|V|} \exp \left(u_{j}^{T} \hat{v}\right)
\end{aligned}
$$
事实上这里的loss function就是上面的交叉熵的进一步推导。我们可以采用**随机梯度下降**SGD的方式来进行模型的求解和参数的更新。



### Skip-gram模型

Skip-gram模型是一种给定中心单词来预测上下文中各个位置的不同单词出现概率的模型，模型的工作原理可以用如下几个步骤概括：

1.  生成中心单词的one-hot向量$x\in \mathbb{R}^{|V|}$作为输入
2.  得到输入单词的嵌入向量$v_c=\mathcal{V}x$并计算分数向量$z=\mathcal{U}{v_c}$
3.  将分数向量用softmax转化成对应的概率分布$\hat{y}=\mathrm{softmax}(z)\in \mathbb R^{|V|}$，则
    $$\left(\hat{y}^{(c-m)}, \ldots, \hat{y}^{(c-1)}, \hat{y}^{(c+1)}, \ldots, \hat{y}^{(c+m)} \in \mathbb{R}^{|V|}\right)$$
是预测出的对应位置中出现的单词的概率分布，每个概率向量的维度都和词汇表的大小一样
    
4.  根据概率分布生成对应的one-hot向量，作为最终的预测结果，这里就是将概率最大的那个维度对应的单词作为结果。

#### 参数的求解和模型的优化

与CBOW类似，可以采用类似的方法来优化模型的参数，首先Skip-gram模型是通过中心单词c来预测上下文单词o的出现概率，这一过程可以表示为：
$$
P(o|c)=\frac{\exp(u_o^Tv_c)}{\sum_{w\in V}\exp(u_w^Tv_c)}
$$
而其损失函数可以表示为：
$$
J=-\sum_{w\in V}y_w\log (\hat y_w)=-\log P(O=o|C=c)=-u_o^Tv_c+\log (\sum_{w\in V}\exp(u_w^Tv_c))
$$
这样一来损失函数J对于词向量$u,v$的导数分别可以用如下方式来计算，首先来看中心词向量v的导数：
$$
\begin{aligned}
        \frac{\partial J}{\partial v_{c}} &=-\frac{\partial\left(u_{o}^{T} v_{c}\right)}{\partial v_{c}}+\frac{\partial\left(\log \left(\sum_{w} \exp \left(u_{w}^{T} v_{c}\right)\right)\right)}{\partial v_{c}} \\
        &=-u_{o}+\frac{1}{\sum_{w} \exp \left(u_{w}^{T} v_{c}\right)} \frac{\partial\left(\sum_{w} \exp \left(u_{w}^{T} v_{c}\right)\right)}{\partial v_{c}} \\
        &=-u_{o}+\sum_{w} \frac{\exp \left(u_{w}^{T} v_{c}\right) u_{w}}{\sum_{w} \exp \left(u_{w}^{T} v_{c}\right)} \\
        &=-u_{o}+\sum_{w} P(O=w \mid C=c) u_{w} \\
        &=-u_{o}+\sum_{w} \hat{y}_{w} u_{w} \\
        &=-u_{o}+u_{\text {new }} \\
        &=U^{T}(\hat{y}-y)
    \end{aligned}
$$
我们首先要搞清楚真实的结果y是一个one-hot向量，所以上述步骤是可以成立的，而对于向量$u_w$，则要分情况讨论，当w=o的时候，有
$$
\begin{aligned}
        \frac{\partial J}{\partial u_{w}} &=-\frac{\partial\left(u_{o}^{T} v_{c}\right)}{\partial u_{w}}+\frac{\partial\left(\log \left(\exp \left(u_{w}^{T} v_{c}\right)\right)\right)}{\partial u_{w}} \\
        &=-v_{c}+\frac{1}{\sum_{w} \exp \left(u_{w}^{T} v_{c}\right)} \frac{\partial\left(\exp \left(u_{w}^{T} v_{c}\right)\right)}{\partial u_{w}} \\
        &=-v_{c}+\frac{\exp \left(u_{w}^{T} v_{c}\right) v_{c}}{\sum_{w} \exp \left(u_{w}^{T} v_{c}\right)} \\
        &=-v_{c}+p(O=w \mid C=c) v_{c} \\
        &=-v_{c}+\hat{y}_{w=o} v_{c}
        \end{aligned}
$$
而当w是其他值的时候，有：
$$
\frac{\partial J}{\partial u_{w}}=0+\sum_{w\not = o }P(O=w|C=c)v_c=\sum_{w\not= o}\hat y_{w}v_c
$$
考虑到真实结果y是一个one-hot向量，上面两个式子又可以合并为：
$$
\frac{\partial J}{\partial U}=v_c^T(\hat y- y)
$$
- 上面的这些公式推导在CS224N的编程作业中可能会用到，当然作业中是用numpy手动实现梯度的计算，一般情况这些东西都用和`loss.backward()`和`optim.step()`来解决
- 不过初学的时候推一推公式还是很有意思的，当然我现在已经忘记光了

### 负采样Negative Sampling

上面提到的算法中，时间复杂度基本都是$O(|V|)$的(比如softmax计算中，会消耗大量的算力)，也就是词汇表的大小，而这个规模往往是比较大的，因此我们可以想办法逼近这个值。

在每一次的迭代中，不是遍历整个词汇表而是进行一些负采样操作，我们可以从一个噪声分布中"采样"，其概率与词汇表频率的顺序匹配。

负采样可以理解为单词和短语及其构成的分布式表示，比如在skip-gram模型中，假设一对单词和上下文$(w,c)$，我们用$P(D|w,c),D\in \left\{0,1\right\}$来表示**一对单词和上下文是否属于训练集中出现过的内容**，首先可以用sigmoid函数来定义这个概率：
$$
P(D=1 \mid w, c, \theta)=\sigma\left(v_{c}^{T} v_{w}\right)=\frac{1}{1+e^{-v_{c}^{T} v_{w}}}
$$
这里的v就是上面提到的嵌入向量，然后我们可以构建一个新的目标函数来试图最大化一个单词和上下文对属于训练文本的概率，也就是采用**极大似然法来估计参数**，这里我们将需要求的参数$\mathcal{V},\mathcal{U}$记作$\theta$，则极大似然估计的目标可以表示为：
$$
\begin{aligned}
        \theta &=\arg\max_{\theta} \prod_{(w, c) \in D} P(D=1 \mid w, c, \theta) \prod_{(w, c) \in \tilde{D}} P(D=0 \mid w, c, \theta) \\
        &=\arg\max_{\theta} \prod_{(w, c) \in D} P(D=1 \mid w, c, \theta) \prod_{(w, c) \in \tilde{D}}(1-P(D=1 \mid w, c, \theta)) \\
        &=\arg\max_{\theta} \sum_{(w, c) \in D} \log P(D=1 \mid w, c, \theta)+\sum_{(w, c) \in \tilde{D}} \log (1-P(D=1 \mid w, c, \theta)) \\
        &=\arg\max_{\theta} \sum_{(w, c) \in D} \log \frac{1}{1+\exp \left(-u_{w}^{T} v_{c}\right)}+\sum_{(w, c) \in \tilde{D}} \log \left(1-\frac{1}{1+\exp \left(-u_{w}^{T} v_{c}\right)}\right) \\
        &=\arg\max_{\theta} \sum_{(w, c) \in D} \log \frac{1}{1+\exp \left(-u_{w}^{T} v_{c}\right)}+\sum_{(w, c) \in \tilde{D}} \log \left(\frac{1}{1+\exp \left(u_{w}^{T} v_{c}\right)}\right)
    \end{aligned}
$$
因此可以损失函数可以等价地采用如下形式，相对的，我们需要求的就是损失函数的最小值：

$$
J=-\sum_{(w, c) \in D} \log \frac{1}{1+\exp \left(-u_{w}^{T} v_{c}\right)}-\sum_{(w, c) \in \tilde{D}} \log \frac{1}{1+\exp \left(u_{w}^{T} v_{c}\right)}
$$

这里的集合$\tilde{D}$ 表示负采样集合，也就是一系列**不属于训练文本的中心词和上下文的对构成的一个集合**，我们可以通过根据训练文本随机生成的方式来得到负采样集合。这样一来CBOW的损失函数就变成了：

$$
J=-\log \sigma\left(u_{c}^{T} \cdot \hat{v}\right)-\sum_{k=1}^{K} \log \sigma\left(-\tilde{u}_{k}^{T} \cdot \hat{v}\right)
$$

而Skip-gram的新损失函数就变成了：
$$
J=-\log \sigma\left(u_{o}^{T} \cdot v_{c}\right)-\sum_{k=1}^{K} \log \sigma\left(-\tilde{u}_{k}^{T} \cdot v_{c}\right)
$$

这里的$\tilde{u_k}$是随机采样产生的K个不是当前中心词的上下文的单词，在skip-gram模型中，梯度可以表示为：
$$
\begin{aligned}
        \frac{\partial J_{neg}}{\partial v_{c}} &=\left(\sigma\left(u_{o}^{T} v_{c}\right)-1\right) u_{o}+\sum_{k=1}^{K}\left(1-\sigma\left(-u_{k}^{T} v_{c}\right)\right) u_{k} \\
        &=\left(\sigma\left(u_{o}^{T} v_{c}\right)-1\right) u_{o}+\sum_{k=1}^{K} \sigma\left(u_{k}^{T} v_{c}\right) u_{k}
        \end{aligned}
$$

而对于$u_o$和负采样向量$\tilde{u_k}$，其梯度是：
$$
\frac{\partial J_{neg}}{\partial u_{o}}=\left(\sigma\left(u_{o}^{T} v_{c}\right)-1\right) v_{c}
$$

$$
\frac{\partial J_{neg}}{\partial u_{k}}=\sum_{k=1}^{K}\left(1-\sigma\left(-u_{k}^{T} v_{c}\right)\right) v_{c}=\sum_{k=1}^{K}\left(\sigma\left(u_{k}^{T} v_{c}\right)\right) v_{c}
$$

全局词向量GloVe
-----------------------

### 已有的单词表示方法

到现在为止已经介绍过的单词表示方法主要分为两种，一种是基于次数统计和矩阵分解的传统机器学习方法，这类方法可以获取**全局**的统计信息，并很好的捕捉单词之间的相似性，但是在单词类比之类的任务重表现较差，比较经典的有潜在语义分析(LSA)等等

另一类是上面提到的基于\"浅窗口\"(shallow window based)的方法，比如CBOW和Skip-gram，通过局部的上下文来进行上下文或者中心单词的预测，这样的方法可以捕捉到复杂的语言模式和上下文信息，但是**对全局的统计信息知之甚少**，缺少\"大局观\"，这也是这些模型的缺点。

而GloVe则使用一些全局的统计信息，比如共生矩阵，并使用最小二乘损失函数来预测一个单词出现在一段上下文中的概率，并且在单词类比的任务上取得了SOTA的效果。

### GloVe算法

首先用$X$来表示训练集的共生矩阵，而$X_{ij}$表示单词i的上下文中出现单词j的次数，用$X_i$来表示矩阵中第i行的和则这样一来：
$$
P_{ij}=P(w_j|w_i)=\frac{X_{ij}}{X_i}
$$

可以表示单词j在单词i的上下文中出现的概率。我们在Skip-gram中，用Softmax来计算单词j出现在i的上下文中的概率：

$$
Q_{i j}=\frac{\exp \left(\vec{u}_{j}^{T} \vec{v}_{i}\right)}{\sum_{w=1}^{W} \exp \left(\vec{u}_{w}^{T} \vec{v}_{i}\right)}
$$

而训练的过程中用到的损失函数如下所示，又因为上下文的关系在样本中可能会多次出现，因此按照如下方式进行变形：

$$
J=-\sum_{i \in \text { corpus }(i)} \sum_{j \in \text { context(i)}}{\log Q_{i j}}=\sum_{i=1}^W\sum_{j=1}^WX_{ij}\log Q_{i j}
$$

交叉熵损失的一个显著缺点是它要求分布Q被适当地标准化，因此可以换一种方式，也就是使用基于最小二乘的目标函数来进行优化求解，就可以不用进行标准化了：

$$
\hat{J}=\sum_{i=1}^{W} \sum_{j=1}^{W} X_{i}\left(\hat{P}_{i j}-\hat{Q}_{i j}\right)^{2}=\sum_{i=1}^{W} \sum_{j=1}^{W} X_{i}\left(X_{ij}-\exp(\vec{u}_{j}^{T})\vec{v}_{i}\right)^2
$$

但是这样子的目标函数又带来了一个新的问题，那就是 $X_{ij}$ 往往是一个比较大的数字，这会对计算量造成很大的影响，一个有效的方法是取对数：
$$
\begin{aligned}
        \hat{J} &=\sum_{i=1}^{W} \sum_{j=1}^{W} X_{i}\left(\log (\hat{P})_{i j}-\log \left(\hat{Q}_{i j}\right)\right)^{2} \\
        &=\sum_{i=1}^{W} \sum_{i=1}^{W} X_{i}\left(\vec{u}_{j}^{T} \vec{v}_{i}-\log X_{i j}\right)^{2}
    \end{aligned}
$$


### 结论

GloVe模型通过只训练共生矩阵中的非零元素充分利用了训练样本的全局信息，并且提供了一个拥有比较有意义的子结构的向量空间，在单词类比的任务中表现优于传统的Word2Vec

词向量的评估
------------

目前已经学习了一系列将单词用词向量来表示的方法，接下来的这一部分主要探讨如何评估生成的词向量的质量的好坏。

### 内部评估

内部评价(Intrinsic Evaluation)是对生成的一系列词嵌入向量，用一些子任务来验证嵌入向量的性能优劣，并且一般这样的子任务是比较容易完成的，这样可以快速反馈词向量训练的结果。

### 外部评估

外部评估是用一些实际中的任务来评估当前训练获得的词向量的性能，但是在优化一个表现不佳的外部评估系统的时候我们无法确定是哪个特定的子系统产生的错误，这就激发了对内在评估的需求。
