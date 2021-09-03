# PyTorch学习笔记1：PyTorch的pipeline

​		![pytorch](https://pytorch.apachecn.org/docs/img/logo.svg)

​		PyTorch是一个开源的深度学习框架，其主要提供的功能是使用GPU加速张量的计算(和numpy非常类似)并且提供了自动微分等功能，因此可以用于构建深度学习模型，只需要写好前向传播的代码，就可以在训练的过程中由PyToch自动进行微分和反向传播。整个PyTorch框架使用的pipeline大致包括数据读取，数据加载，模型搭建，模型训练和参数更新，模型测试这样几个环节，而每个步骤中PyTorch都提供了一些相应的API来简化模型的搭建。

## 数据的读取和加载

​	  在`torch.utils.data`中提供了一个Dataset类和DataLoader类用来进行数据的读取和加载。我们需要先把原数据转化成一个Dataset类，然后再将Dataset作为参数来构造一个DataLoader，DataLoader是一个数据加载器，可以在每次返回一个batch的数据用于训练。

​	  而Dataset是一个抽象类，我们为专门的问题的专门数据集定义一个数据集类的时候要继承Dataset这个类，并且在`__init__`方法里定义数据集的读入的格式，然后在`__getitem__`方法里面定义数据的索引方式，在`__len__`方法里定义数据集的大小用于后期进行遍历，这实际上都是对Dataset抽象类中三个对应方法的具体实现。然后可以将一个继承了Dataset的子类对象和batch size用作参数来构造一个DataLoader，就可以分批获取数据用于训练，可以使用迭代器来获取一个batch的数据。

​	  同时PyTorch的torchvision等库提供了很多现成的数据集和模型，可以直接调用相关的API使用。

## 构建PyTorch模型

​	  PyTorch中的`torch.nn`提供了很多神经网络层的API，基本常见的不常见的神经网络层都有，这些神经网络层相关的类都继承自基类`nn.Module`，这个类的部分定义如下所示：

```python

class Module(object):
    def __init__(self):
    def forward(self, *input):
 
    def add_module(self, name, module):
    def cuda(self, device=None):
    def cpu(self):
    def __call__(self, *input, **kwargs):
    def parameters(self, recurse=True):
    def named_parameters(self, prefix='', recurse=True):
    def children(self):
    def named_children(self):
    def modules(self):  
    def named_modules(self, memo=None, prefix=''):
    def train(self, mode=True):
    def eval(self):
    def zero_grad(self):
    def __repr__(self):
    def __dir__(self):
```

这里的forward方法可以定义前向传播的过程，而在构建深度学习模型的时候，我们需要先定义一个自己的模型类，并且让它继承`nn.Module`基类，在init方法中定义一系列所需要的神经网络层，然后根据模型前向传播的方式，来实现其forward方法，就算是搭建好了一个模型。

- 一般来说有可学习参数的层必须要定义在init构造方法中，而没有可学习参数的(比如ReLU层等)可以直接在forward方法中写出，这个时候可以用`nn.Functional`中的各种函数代替
- 定义模型所需要的各种层的时候可以用`nn.Sequential`将若干个层包装成一个对象

下面是一个简单的LSTM

```python
class NaiveLSTM(nn.Module):
    def __init__(self, max_words, emb_size, hid_size, dropout):
        super(NaiveLSTM, self).__init__()
        self.max_words = max_words
        self.emb_size = emb_size
        self.hid_size = hid_size
        self.dropout = dropout
        self.Embedding = nn.Embedding(self.max_words, self.emb_size)
        self.LSTM = nn.LSTM(self.emb_size, self.hid_size, num_layers=2,
                            batch_first=True, bidirectional=True)  # 2层双向LSTM
        self.dp = nn.Dropout(self.dropout)
        self.fc1 = nn.Linear(self.hid_size * 2, self.hid_size)
        self.fc2 = nn.Linear(self.hid_size, 2)

    def forward(self, x):
        """
        input : [bs, max_len]
        output: [bs, 2]
        """
        x = self.Embedding(x)  # [bs, ml, emb_size]
        x = self.dp(x)
        x, _ = self.LSTM(x)  # [bs, ml, 2*hid_size]
        x = self.dp(x)
        x = F.relu(self.fc1(x))  # [bs, ml, hid_size]
        x = F.avg_pool2d(x, (x.shape[1], 1)).squeeze()  # [bs, 1, hid_size] => [bs, hid_size]
        out = self.fc2(x)  # [bs, 2]
        return out  # [bs, 2]
```



## 模型的训练

​	  我们定义好模型之后就可以实例化出一个新的对象，比如上面这个LSTM就可以用`model=NaiveLSTM(MAX_WORDS, EMB_SIZE, HID_SIZE, DROPOUT).to(DEVICE)`的方式先创建一个模型的对象，我们需要定义一个优化器并将模型的参数绑定在这个优化器上，并且需要定义一个有效的损失函数`loss_fn`，然后用DataLoader生成的batch数据进行前向传播，并得到输出的结果pred，用输出结果和真实的label计算loss，然后进行反向传播，大致的过程如下：

```python
import torch
import torch.nn as nn

model = NaiveLSTM(MAX_WORDS, EMB_SIZE, HID_SIZE, DROPOUT).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters())
loss_function = nn.MSELoss()
for batch_idx, (x, y) in enumerate(train_data_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        y_ = model(x)
        loss = criterion(y_, y)  # 得到loss函数
        loss.backward() # 反向传播
        optimizer.step() # 使用优化器进行参数更新

```

在训练的时候需要指定轮数epoch和batch size等一系列超参数，训练完成之后还需要根据测试集对结果进行评估。整个PyTorch训练深度学习模型的大致过程就是这样，具体的还需要更多的实践，作为学习PyTorch的第一步，就先讲到这里为止。



## CUDA加速

### GPU和CPU的异构计算

​	  这一部分在另一个目录intern下有关于GPU/CUDA更详细的叙述，总的来说GPU体系结构的设计对计算密集型任务进行了一定的特化，因此对计算密集型的任务处理效率特别高，而对于一些复杂的逻辑处理还是应该依赖CPU，因此也就形成了CPU-GPU的异构计算体系。本文主要通过一些实验来探究PyTorch中调用GPU带来的计算加速效果。

### PyTorch如何调用GPU

​	  PyTorch集成了CUDA相关的功能，当前计算机是否可以调用GPU可以用`torch.cuda.is_available()`方法判断，同时PyTorch中的张量可以用`cuda()`方法从CPU中转移到GPU中进行计算，也可以用`to(device)`方法在CPU和GPU之间任意地切换，被转移到GPU中的张量相关的计算就会在GPU中进行。









