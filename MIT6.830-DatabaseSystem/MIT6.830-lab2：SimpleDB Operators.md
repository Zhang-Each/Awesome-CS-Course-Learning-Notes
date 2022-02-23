# MIT6.830-lab2：SimpleDB Operators

> MIT6.830 Database Systems课程Project的第二部分，我们需要实现一些数据库系统中基本的算子和对应的Buffer、Page的管理。我们主要需要实现的操作有Filter, Join,  Aggregate, Insert和Delete

## Project Overview

在6.830的Project2中，我们要实现常见的SQL语句所需要的数据库操作，在Project1中，我们已经实现了一种顺序扫描的操作SeqScan，它的实现方式是利用SimpleDB的Heap File类提供的迭代器来遍历整个数据表文件中的所有元组信息。而在Project2中，我们需要实现几种更复杂的算子，包括：

- Filter，按条件筛选符合的元组
- Join，将两个表中符合条件的元组进行join操作
- Aggregate，按照一定的目标对表中的元组进行聚合运算
- Insert和Delete，插入、删除元组

同时，对于需要修改文件内容的操作(这里只有Insert和Delete会修改已有文件中的内容，其他的都是另外生成新的内容)，我们还需要在Buffer，File和Page三个层面来实现对应的修改操作，使得数据库能够保证一致性。



## Operator的代码实现

### Filter操作与Predicate

Filter操作是按照一定的条件对数据表中的元组进行过滤，这个判断的过程在SimpleDB中被抽象成了一个叫做Predicate的类，这个类在实现的时候，需要用以下三个信息进行初始化，分别是：

- 需要判断的属性值的列号
- 判断的类型(大于，小于，等于，不等于)，在SimpleDB中**被抽象成了一个枚举类Op**
- 判断的基准值，用之前说到过的Field类型来表示

然后Predicate类会提供一个方法filter判断给定元组的某一列的属性值是否满足判断条件，并返回True或者False，然后我们在Filter Operator类中调用Predicate来帮助我们判断就可以。

Filter继承了Operator并需要实现`fetchNext`方法，而Filter的这个方法需要从child那里读取一个个tuple并进行判断，直到找到一个合适的可以返回的元组才能返回，如果child的遍历已经结束了，那么就返回一个null



### Join操作与Join Predicate

Join操作是对两个数据表进行操作，将分别来自于两个表的，满足一定条件的元组a和b合成一个新的元组c，并且c的所有属性是a和b汇总得到的，比如a元组的是$(a_1,a_2,a_3)$， b元组是$(b_1,b_2)$ 且ab满足join的判断条件，那么新生成的c就是$(a_1,a_2,a_2,b_1,b_2)$ 

