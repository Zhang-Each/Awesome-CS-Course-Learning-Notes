# MLC02-Tensor Program

> Tianqi Chen开设的暑期课程，主题是机器学习编译，课程网站：https://mlc.ai/summer22/，这是第二节课张量程序抽象的内容

## 元张量函数Primitive Tensor Function

元张量函数(Primitive Tensor Function)是指张量计算过程中的最基本单元，比如Linear层，ReLU层，SoftMax层等等都可以看成是元张量函数。元张量函数可以用多种不同的抽象来表示，比如一个add函数就有这样几种不同的抽象：

![image-20220628142020298](https://raw.githubusercontent.com/Zhang-Each/Image-Bed/main/img/image-20220628142020298.png)

在PyTorch中，add就是一个可以直接使用的函数，而add函数在Python层面的实现方式就如第二段代码所示，PyTorch的底层实现是用C++完成的，在C++里面，add函数就会变成第三种形式。许多机器学习框架都提供**机器学习模型的编译过程**，以将元张量函数变换为更加专门的、**针对特定工作和部署环境的函数**。

![image-20220628142543491](https://raw.githubusercontent.com/Zhang-Each/Image-Bed/main/img/image-20220628142543491.png)

## 张量程序抽象

为了有效地对元张量函数进行变换，我们需要建立有效的抽象来表示这些函数。一个典型的元张量函数的抽象需要包括这样几个部分：

- 存储数据的多维数组(Buffer)
- 用于进行张量计算的for循环
- 具体的计算逻辑

![image-20220628234511216](https://raw.githubusercontent.com/Zhang-Each/Image-Bed/main/img/image-20220628234511216.png)

张量程序抽象有一个重要的性质，那就是它能够被一系列程序进行转换，我们可以通过一系列操作(比如循环拆分、并行化、向量化)讲原始的程序变换成新的程序。

![image-20220628234643911](https://raw.githubusercontent.com/Zhang-Each/Image-Bed/main/img/image-20220628234643911.png)

## 张量程序变换实践

这个在课程主页里有相关的代码演示。