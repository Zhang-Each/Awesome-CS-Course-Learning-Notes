词向量Word Vector
=================

词向量也就是用向量来表示一个单词，但是语言的单词数量可能是非常庞大的，比如英语就有各类单词词组约1300万个，我们需要将这些单词全部编码到一个N维度的向量中去，向量的每个维度可以编码一些语义或者单词的特征。词向量也可以叫做词嵌入$Word Embedding$

one-hot向量
-----------

一种非常不负责任的编码方式就是one-hot向量，这种方法根据当前样本数据的词汇表的大小$|V|$，用一系列维度为$|V|$的0-1向量来代表单词，词汇表中出现的每个单词$m$有一个对应的维度$v_m$，每个单词对应的词向量中，该维度的值是1，其他的维度都是0，这种方式非常简单粗暴，但是问题也很明显，这样的编码是非常稀疏的，词向量中的大部分内容都是无效信息0，并且词向量的规模也非常大，是对计算资源的严重浪费。

基于奇异值分解SVD的词向量
-------------------------

### 奇异值分解SVD

根据我们仅存的一点线性代数的知识，我们知道矩阵可以分解成一系列特征向量和特征值，但是除此之外矩阵还可以进行奇异值分解(Singular
Value Decomposition,
SVD)是将矩阵分解成一系列奇异向量和奇异值，这种方法可以使我们得到一些和特征分解类似的信息，**但是相比于特征分解，奇异分解适用范围更广，所有的实矩阵都有一个奇异值分解但不一定有特征分解(特征分解必须要是方阵)**。
在特征分解中我们可以求出一系列特征向量使其构成一个矩阵$V$和一系列特征值构成对角矩阵$\mathrm{diag}(\lambda)$，这样一来一个矩阵就可以分解成：
$$A=V\mathrm{diag}(\lambda) V^{-1}$$
而在奇异值分解中我们可以类似地将矩阵分解成三个矩阵的乘积：
$$A=UDV^{-1}$$

这里我们可以假设A是一个$m\times n$维的矩阵那么U是一个$m\times m$的矩阵，D是一个$m\times n$的矩阵，V是一个$n\times n$的矩阵并且U和V都是正交矩阵，而D是一个对角矩阵。

对角矩阵D的对角线上的元素就是矩阵A的奇异值，U和V分别被称为左右奇异向量，事实上这种分解方式可以理解为：D就是$AA^T$特征值的平方根，而U和V分别是$AA^T$和$A^TA$的特征向量，实际上就是我们将任意一个实矩阵通过一定的变换变成了一个方阵，然后对方阵进行奇异值分解反过来表示原本矩阵的一些特征。

因此我们可以适用奇异值分解的办法来提取一个单词的嵌入向量，但是我们首先需要将数据集中海量的单词汇总成一个矩阵X并对该矩阵进行奇异值分解，然后可以将分解得到的U作为所有单词的嵌入向量，而生成矩阵X的方法有多种选择。

### 单词文档矩阵Word-Document Matrix

我们可以假设相关的单词总会在同一篇文档中出现，利用这一假设来建立一个单词文档矩阵，其中$X_{ij}=1$表示单词$w_i$出现在了文档$d_j$中，否则就是0，这样一来矩阵的规模就是$|V|\times M$，规模是非常庞大的。

### 基于窗口的共生矩阵Co-occurrence Matrix

我们可以使用一个固定大小的滑动窗口来截取文档中的单词组合，然后对每一次采样得到的单词进行计数，如果单词a和单词b在一次采样结果中同时出现，就在矩阵X上给它们对应的位置的值+1(相同的单词不考虑)，这样一来可以得到一个大小为$|V|\times |V|$的矩阵并且U和V都是正交矩阵，而D是一个对角矩阵。

### SVD的使用

在获得的矩阵X上使用SVD的时候，最后得到的每个单词的嵌入向量是$|V|$维的，我们可以进行截取，只保留k维：
$$\frac{\sum_{i=1}^k\sigma_i}{\sum_{i=1}^{|V|}\sigma_i}$$
然后用截取得到的k维向量作为每个单词的嵌入向量。

### SVD方法的缺陷与解决

SVD方法还是存在一定的缺陷的，不管用什么方式生成矩阵X，都会面临这样几个问题：

-   计算量较大，对于$m\times n$的矩阵，SVD的时间复杂度是$O(mn^2)$，因此维数高了以后计算量会暴增

-   生成的矩阵要插入新单词或者新文档的内容时比较麻烦，矩阵的规模很容易变化

-   生成的矩阵往往比较稀疏，因为大部分单词和单词或者单词和文档之间可能没什么关系

面对这些问题，可以采取的解决措施有：

-   不考虑一些功能性的单词比如a，an，the，it等等

-   使用一个倾斜的窗口，也就是在生成共生矩阵的时候考虑单词相距的距离

Word2vec
--------

### 基于迭代的模型

基于SVD的方法实际上考虑的都是单词的全局特征，因此计算代价和效果都不太好，因此我们可以尝试建立一个基于迭代的模型并根据单词在上下文中出现的概率来进行编码。

另一种思路是设计一个参数是词向量的模型，然后针对一个目标进行模型的训练，在每一次的迭代中计算错误率或者loss函数并据此来更新模型的参数，也就是词向量，最后就可以得到一系列结果作为单词的嵌入向量。常见的方法包括bag-of-words
(CBOW)和skip-gram，其中CBOW的目标是根据上下文来预测中心的单词，而skip-gram则是根据中心单词来预测上下文中的各种单词出现的概率。而这些模型常用的训练方法有：

-   负采样negative
    sampling：判断两个单词是不是一对上下文与目标词，如果是一对，则是正样本，如果不是一对，则是负样本，而采样得到一个上下文词和一个目标词，生成一个正样本，用与正样本相同的上下文词，再在字典中随机选择一个单词，这就是负采样

-   层级化(hierarchical)的softmax

### 语言模型Language Model

语言模型可以给一系列单词序列指定一个概率来判断它是不是可以作为一个完整的句子输出，一般完整性强的，语义通顺的句子被赋予的概率会更大，而无法组成句子的一系列单词可能就会计算出非常小的概率，语言模型用公式来表示就是：
$$P\left(w_{1}, w_{2}, \cdots, w_{n}\right)$$
如果我们假设每个单词在每个位置出现的概率是独立的话，这个表达式就会变成：
$$P\left(w_{1}, w_{2}, \cdots, w_{n}\right)=\prod_{i=1}^nP(w_{i})$$
但这显然是不可能的，因为单词的出现并不是不受约束的，而是和其周围的单词(也就是上下文)有一定的联系，这也是NLP中一个非常重要的观点，那就是**分布式语义(Distributional
semantics)：一个词语的意思是由附近的单词决定的，一个单词的上下文就是"附近的单词"，可以通过一个定长的滑动窗口来捕捉每个单词对应的上下文和语义**。

Bigram model认为单词出现的概率取决于它的上一个单词，即：
$$P\left(w_{1}, w_{2}, \cdots, w_{n}\right)=\prod_{i=2}^nP(w_{i}|w_{i-1})$$
但显然这也是一种非常的的模型，因此我们需要寻找更合适的语言模型。

### 连续词袋模型(CBOW)

连续词袋模型Continuous Bag of Words
Model是一种根据上下文单词来预测中心单词的模型，对于每一个单词我们需要训练出两个向量u和v分别代表该单词作为中心单词和作为上下文单词的时候的嵌入向量(一种代表输出结果，一种代表输入)

因此可以定义两个矩阵$\mathcal{V} \in \mathbb{R}^{n \times|V|}$ 和
$\mathcal{U} \in \mathbb{R}^{|V| \times n}$，其中n代表了嵌入空间的维度，矩阵$\mathcal{V}$的每一列代表了词汇表中一个单词作为上下文单词时候的嵌入向量，用$v_i$表示，而矩阵$\mathcal{U}$表示的是输出的结果，每一行代表了一个单词的嵌入向量，用$u_j$表示。

#### CBOW模型的工作原理

1.  首先根据输入的句子生成一系列one-hot向量，假设要预测的位置是c，上下文的宽度各为m，则生成的一系列向量可以表示为：
    $$\left(x^{(c-m)}, \ldots, x^{(c-1)}, x^{(c+1)}, \ldots, x^{(c+m)} \in \mathbb{R}^{|V|}\right)$$

2.  计算得到上下文单词的嵌入向量： $$v_j=\mathcal{V}x^{(j)}$$

3.  求出这些向量的均值：
    $$\hat{v}=\frac{v_{c-m}+v_{c-m+1}+\ldots+v_{c+m}}{2 m} \in \mathbb{R}^{n}$$

4.  生成一个分数向量$z=\mathcal{U}\hat{v}$，当相似向量的点积较大时，会将相似的词推到一起，以获得较高的分数

5.  用softmax函数将分数向量转化成概率分布：$\hat{y}=\mathrm{softmax}(z)\in \mathbb R^{|V|}$

6.  根据概率分布$\hat{y}$得到最终的结果$y$(是一个one-hot向量的形式)，也就是最有可能出现在中心位置的单词

因此现在的问题就变成了，我们应该如何学习到两个最关键的矩阵$\mathcal{V}$和$\mathcal{U}$，这也是CBOW最关键的一个问题，这里可以采用信息论中的交叉熵来定义一个目标函数：
$$H(\hat{y}, y)=-\sum_{j=1}^{|V|} y_{j} \log \left(\hat{y}_{j}\right)=-y_{i} \log \left(\hat{y}_{i}\right)$$
因为概率的真实值$y$是一个one-hot向量，只有一个维度是1其他的都是0，因此可以进行这样的化简。

#### 损失函数与优化方式

我们可以定义如下形式的损失函数并进行优化： $$\begin{aligned}
        \operatorname{minimize} J &=-\log P\left(w_{c} \mid w_{c-m}, \ldots, w_{c-1}, w_{c+1}, \ldots, w_{c+m}\right) \\
        &=-\log P\left(u_{c} \mid \hat{v}\right) \\
        &=-\log \frac{\exp \left(u_{c}^{T} \hat{v}\right)}{\sum_{j=1}^{|V|} \exp \left(u_{j}^{T} \hat{v}\right)} \\
        &=-u_{c}^{T} \hat{v}+\log \sum_{j=1}^{|V|} \exp \left(u_{j}^{T} \hat{v}\right)
    \end{aligned}$$
这里我们可以采用随机梯度下降SGD的方式来进行模型的求解和参数的更新。

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
$$P(o|c)=\frac{\exp(u_o^Tv_c)}{\sum_{w\in V}\exp(u_w^Tv_c)}$$
而其损失函数可以表示为：
$$J=-\sum_{w\in V}y_w\log (\hat y_w)=-\log P(O=o|C=c)=-u_o^Tv_c+\log (\sum_{w\in V}\exp(u_w^Tv_c))$$
这样一来损失函数J对于词向量$u,v$的导数分别可以用如下方式来计算，首先来看中心词向量v的导数：
$$\begin{aligned}
        \frac{\partial J}{\partial v_{c}} &=-\frac{\partial\left(u_{o}^{T} v_{c}\right)}{\partial v_{c}}+\frac{\partial\left(\log \left(\sum_{w} \exp \left(u_{w}^{T} v_{c}\right)\right)\right)}{\partial v_{c}} \\
        &=-u_{o}+\frac{1}{\sum_{w} \exp \left(u_{w}^{T} v_{c}\right)} \frac{\partial\left(\sum_{w} \exp \left(u_{w}^{T} v_{c}\right)\right)}{\partial v_{c}} \\
        &=-u_{o}+\sum_{w} \frac{\exp \left(u_{w}^{T} v_{c}\right) u_{w}}{\sum_{w} \exp \left(u_{w}^{T} v_{c}\right)} \\
        &=-u_{o}+\sum_{w} P(O=w \mid C=c) u_{w} \\
        &=-u_{o}+\sum_{w} \hat{y}_{w} u_{w} \\
        &=-u_{o}+u_{\text {new }} \\
        &=U^{T}(\hat{y}-y)
    \end{aligned}$$
我们首先要搞清楚真实的结果y是一个one-hot向量，所以上述步骤是可以成立的，而对于向量$u_w$，则要分情况讨论，当w=o的时候，有
$$\begin{aligned}
        \frac{\partial J}{\partial u_{w}} &=-\frac{\partial\left(u_{o}^{T} v_{c}\right)}{\partial u_{w}}+\frac{\partial\left(\log \left(\exp \left(u_{w}^{T} v_{c}\right)\right)\right)}{\partial u_{w}} \\
        &=-v_{c}+\frac{1}{\sum_{w} \exp \left(u_{w}^{T} v_{c}\right)} \frac{\partial\left(\exp \left(u_{w}^{T} v_{c}\right)\right)}{\partial u_{w}} \\
        &=-v_{c}+\frac{\exp \left(u_{w}^{T} v_{c}\right) v_{c}}{\sum_{w} \exp \left(u_{w}^{T} v_{c}\right)} \\
        &=-v_{c}+p(O=w \mid C=c) v_{c} \\
        &=-v_{c}+\hat{y}_{w=o} v_{c}
        \end{aligned}$$ 而当w是其他值的时候，有：
$$\frac{\partial J}{\partial u_{w}}=0+\sum_{w\not = o }P(O=w|C=c)v_c=\sum_{w\not= o}\hat y_{w}v_c$$
考虑到真实结果y是一个one-hot向量，上面两个式子又可以合并为：
$$\frac{\partial J}{\partial U}=v_c^T(\hat y- y)$$

### 负采样Negative Sampling

上面提到的算法中，时间复杂度基本都是$O(|V|)$的(比如softmax计算中，会消耗大量的算力)，也就是词汇表的大小，而这个规模往往是比较大的，
因此我们可以想办法逼近这个值。在每一次的迭代中，不是遍历整个词汇表而是进行一些负采样操作，我们可以从一个噪声分布中"采样"，
其概率与词汇表频率的顺序匹配。

负采样可以理解为单词和短语及其构成的分布式表示，比如在skip-gram模型中，假设一对单词和上下文$(w,c)$，我们用$P(D|w,c),D\in \left\{0,1\right\}$来表示**一对单词和上下文是否属于训练集中出现过的内容**，首先可以用sigmoid函数来定义这个概率：
$$P(D=1 \mid w, c, \theta)=\sigma\left(v_{c}^{T} v_{w}\right)=\frac{1}{1+e^{\left(-v_{c}^{T} v_{w}\right)}}$$
这里的v就是上面提到的嵌入向量，然后我们可以构建一个新的目标函数来试图最大化一个单词和上下文对属于训练文本的概率，也就是采用极大似然法来估计参数，这里我们将需要求的参数$\mathcal{V},\mathcal{U}$记作$\theta$，则极大似然估计的目标可以表示为：
$$\begin{aligned}
        \theta &=\underset{\theta}{\operatorname{argmax}} \prod_{(w, c) \in D} P(D=1 \mid w, c, \theta) \prod_{(w, c) \in \tilde{D}} P(D=0 \mid w, c, \theta) \\
        &=\underset{\theta}{\operatorname{argmax}} \prod_{(w, c) \in D} P(D=1 \mid w, c, \theta) \prod_{(w, c) \in \tilde{D}}(1-P(D=1 \mid w, c, \theta)) \\
        &=\underset{\theta}{\operatorname{argmax}} \sum_{(w, c) \in D} \log P(D=1 \mid w, c, \theta)+\sum_{(w, c) \in \tilde{D}} \log (1-P(D=1 \mid w, c, \theta)) \\
        &=\underset{\theta}{\operatorname{argmax}} \sum_{(w, c) \in D} \log \frac{1}{1+\exp \left(-u_{w}^{T} v_{c}\right)}+\sum_{(w, c) \in \tilde{D}} \log \left(1-\frac{1}{1+\exp \left(-u_{w}^{T} v_{c}\right)}\right) \\
        &=\underset{\theta}{\operatorname{argmax}} \sum_{(w, c) \in D} \log \frac{1}{1+\exp \left(-u_{w}^{T} v_{c}\right)}+\sum_{(w, c) \in \tilde{D}} \log \left(\frac{1}{1+\exp \left(u_{w}^{T} v_{c}\right)}\right)
    \end{aligned}$$
因此可以损失函数可以等价地采用如下形式，相对的，我们需要求的就是损失函数的最小值：
$$J=-\sum_{(w, c) \in D} \log \frac{1}{1+\exp \left(-u_{w}^{T} v_{c}\right)}-\sum_{(w, c) \in \tilde{D}} \log \left(\frac{1}{1+\exp \left(u_{w}^{T} v_{c}\right)}\right)$$
这里的集合$\tilde{D}$表示负采样集合，也就是一系列不属于训练文本的中心词和上下文的对构成的一个集合，我们可以通过根据训练文本随机生成的方式来得到负采样集合。这样一来CBOW的损失函数就变成了：
$$J=-\log \sigma\left(u_{c}^{T} \cdot \hat{v}\right)-\sum_{k=1}^{K} \log \sigma\left(-\tilde{u}_{k}^{T} \cdot \hat{v}\right)$$
而Skip-gram的新损失函数就变成了：
$$J=-\log \sigma\left(u_{o}^{T} \cdot v_{c}\right)-\sum_{k=1}^{K} \log \sigma\left(-\tilde{u}_{k}^{T} \cdot v_{c}\right)$$
这里的$\tilde{u_k}$是随机采样产生的K个不是当前中心词的上下文的单词，在skip-gram模型中，梯度可以表示为：
$$\begin{aligned}
        \frac{\partial J_{\text {neg-sample }}}{\partial v_{c}} &=\left(\sigma\left(u_{o}^{T} v_{c}\right)-1\right) u_{o}+\sum_{k=1}^{K}\left(1-\sigma\left(-u_{k}^{T} v_{c}\right)\right) u_{k} \\
        &=\left(\sigma\left(u_{o}^{T} v_{c}\right)-1\right) u_{o}+\sum_{k=1}^{K} \sigma\left(u_{k}^{T} v_{c}\right) u_{k}
        \end{aligned}$$ 而对于$u_o$和负采样向量$\tilde{u_k}$，其梯度是：
$$\frac{\partial J_{\text {neg-sample }}}{\partial u_{o}}=\left(\sigma\left(u_{o}^{T} v_{c}\right)-1\right) v_{c}$$
$$\frac{\partial J_{\text {neg-sample }}}{\partial u_{k}}=\sum_{k=1}^{K}\left(1-\sigma\left(-u_{k}^{T} v_{c}\right)\right) v_{c}=\sum_{k=1}^{K}\left(\sigma\left(u_{k}^{T} v_{c}\right)\right) v_{c}$$



单词表示的全局向量GloVe
-----------------------

### 已有的单词表示方法

到现在为止已经介绍过的单词表示方法主要分为两种，一种是基于次数统计和矩阵分解的传统机器学习方法，这类方法可以获取全局的统计信息，并很好的捕捉单词之间的相似性，但是在单词类比之类的任务重表现较差，比较经典的有潜在语义分析(LSA)等等，另一类是上面提到的基于\"浅窗口\"(shallow
window
based)的方法，比如CBOW和Skip-gram，通过局部的上下文来进行上下文或者中心单词的预测，这样的方法可以捕捉到复杂的语言模式，但是对全局的统计信息知之甚少，缺少\"大局观\"，这也是这些模型的缺点。

而GloVe则使用一些全局的统计信息，比如共生矩阵，并使用最小二乘损失函数来预测一个单词出现在一段上下文中的概率，并且在单词类比的任务上取得了SOTA的效果。

### GloVe算法

首先用$X$来表示训练集的共生矩阵，而$X_{ij}$表示单词i的上下文中出现单词j的次数，用$X_i$来表示矩阵中第i行的和则这样一来：
$$P_{ij}=P(w_j|w_i)=\frac{X_{ij}}{X_i}$$
可以表示单词j在单词i的上下文中出现的概率。

我们在Skip-gram中，用Softmax来计算单词j出现在i的上下文中的概率：
$$Q_{i j}=\frac{\exp \left(\vec{u}_{j}^{T} \vec{v}_{i}\right)}{\sum_{w=1}^{W} \exp \left(\vec{u}_{w}^{T} \vec{v}_{i}\right)}$$
而训练的过程中用到的损失函数如下所示，又因为上下文的关系在样本中可能会多次出现，因此按照如下方式进行变形：
$$J=-\sum_{i \in \text { corpus }(i)} \sum_{j \in \text { context(i)}}{\log Q_{i j}}=\sum_{i=1}^W\sum_{j=1}^WX_{ij}\log Q_{i j}$$
交叉熵损失的一个显著缺点是它要求分布Q被适当地标准化，因此可以换一种方式，也就是使用基于最小二乘的目标函数来进行优化求解，就可以不用进行标准化了：
$$\hat{J}=\sum_{i=1}^{W} \sum_{j=1}^{W} X_{i}\left(\hat{P}_{i j}-\hat{Q}_{i j}\right)^{2}=\sum_{i=1}^{W} \sum_{j=1}^{W} X_{i}\left(X_{ij}-\exp(\vec{u}_{j}^{T})\vec{v}_{i}\right)^2$$
但是这样子的目标函数又带来了一个新的问题，那就是$X_{ij}$往往是一个比较大的数字，这会对计算量造成很大的影响，一个有效的方法是取对数：
$$\begin{aligned}
        \hat{J} &=\sum_{i=1}^{W} \sum_{j=1}^{W} X_{i}\left(\log (\hat{P})_{i j}-\log \left(\hat{Q}_{i j}\right)\right)^{2} \\
        &=\sum_{i=1}^{W} \sum_{i=1}^{W} X_{i}\left(\vec{u}_{j}^{T} \vec{v}_{i}-\log X_{i j}\right)^{2}
    \end{aligned}$$

### 结论

GloVe模型通过只训练共生矩阵中的非零元素充分利用了训练样本的全局信息，并且提供了一个拥有比较有意义的子结构的向量空间，在单词类比的任务中表现优于传统的Word2Vec

词向量的评估
------------

目前已经学习了一系列将单词用词向量来表示的方法，接下来的这一部分主要探讨如何评估生成的词向量的质量的好坏。

### 内部评估

内部评价(Intrinsic
Evaluation)是对生成的一系列词嵌入向量，用一些子任务来验证嵌入向量的性能优劣，并且一般这样的子任务是比较容易完成的，这样可以快速反馈词向量训练的结果。

### 外部评估

外部评估是用一些实际中的任务来评估当前训练获得的词向量的性能，但是在优化一个表现不佳的外部评估系统的时候我们无法确定是哪个特定的子系统产生的错误，这就激发了对内在评估的需求。
