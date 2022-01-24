# CMU15-445-Project0

> 这是CMU的15-445数据库系统课程的Project记录，记录一些写Project时候的想法，感悟和收获，首先是Project0

CMU15-445的课程Project是用C++实现一个关系型数据库Bustub中的关键代码，但是这个数据库好像就是一个底层引擎，并没有前端系统(即SQL解析器)，不过无伤大雅(大概)，总之首先从lab0开始。

lab的初始代码被放在一个GitHub的公开仓库中，首先git clone一下然后按照要求运行一下安装脚本就算配置好环境了，应该说第一步配置环境算是比较简单的，而lab0的主要目的是复习C/C++的基本语法和Cmake等工具的使用方式。

Project0主要是写几个和矩阵相关的类，首先是矩阵类`Matrix`，然后是行矩阵类`RowMatrix`和矩阵运算类`RowMatrixOperations`，这里要实现的内容主要就是：将矩阵首先用一维数组来表示，然后在行矩阵中用二维数组表示一个矩阵，然后实现矩阵的加法和乘法等各种操作。

大部分代码都是没啥含金量的，Project0主要就是带你熟悉一下C++的基本语法和项目管理工具(比如Cmake，GTest的使用)，值得一提的是，第三部分写RowMatrixOperations类的若干函数时用到了智能指针，具体的代码是：

```C++
static std::unique_ptr<RowMatrix<T>> Add(const RowMatrix<T> *matrixA, const RowMatrix<T> *matrixB) {
    // TODO(P0): Add implementation
    if (matrixA->GetRowCount() != matrixB->GetRowCount() || matrixA->GetColumnCount() != matrixB->GetColumnCount()) {
      return std::unique_ptr<RowMatrix<T>>(nullptr);
    }
    int row = matrixA->GetRowCount();
    int col = matrixA->GetColumnCount();
    std::unique_ptr<RowMatrix<T>> result(new RowMatrix<T>(row, col));
    for (int i = 0; i < row; ++i) {
      for (int j = 0; j < col; ++j) {
        result->SetElement(i, j, matrixA->GetElement(i, j) + matrixB->GetElement(i, j));
      }
    }
    return result;
  }
```

这里的参数是两个行矩阵RowMatrix，而返回值的类型是一个RowMatrix的智能指针，我一开始写的时候已经把智能指针的用法忘记掉了，查了一下才知道要怎么样声明一个智能指针。

同时整个项目使用的代码规范是Google的C++ Style，并且提供了check-lint和check-clang-tidy两个小工具对代码进行格式化，当然我个人不是很喜欢这种风格(主要是指空格只有一个space，我比较喜欢空两格)