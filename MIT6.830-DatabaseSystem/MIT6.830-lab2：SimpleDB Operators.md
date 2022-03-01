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

而join操作也需要进行条件判断，和Filter不同的是，这里的判断的参数变成了两个元组，所以SimpleDB设计了一个类Join Predicate，这个Join Predicate的实现方式和Predicate基本上类似，只不过



### Aggregate操作

Aggregate操作是对数据表中的元组按照一定的规则进行聚合，常见的Aggregate操作包括：

- 计数类Count
- 求和类Sum / Avg
- 最值类Max / Min

Aggregate常常和`Group By`一起使用，`Group By`就是聚合的依据，当它指定了某个属性之后，该属性相同的元组会被聚合成一个结果，这时候Aggregate操作返回的结果就是形如`(groupValue, aggregateValue)`的元组，对每个不同的group value分别进行聚合，而如果没有`Group By`关键字，那么返回的结果就只剩一个aggregateValue，在lab2中，我们需要先分别实现两种不同数据类型(Int 和 String)各自的聚合运算，然后合并成SimpleDB的聚合运算符

- 值得注意的是，我们在实现Aggregate运算符的时候，需要自己定义返回结果的TupleDesc(就是上面提到的两种格式，判断依据是有没有进行group)



### Delete和Insert操作

接下来我们还要实现Delete和Insert两种操作，这两种操作和前面的区别在于，Delete和Insert会改变数据表的Page中存储的元组信息，比如Insert会找到一个空的slot并插入元组，Delete会把对应slot上的元组标记为invalid，这样就表示这段位置上的元组已经失效，可以填入其他的元组。

#### 存储系统中的修改操作

为此我们要先实现每个Heap Page上的插入删除操作，然后实现每个Heap File上的插入删除，然后再实现BufferPool中的插入删除。这和我们在lab1中构建SimpleDB的存储体系的顺序是一样的。同时我们也会发现，之前定义的对bitMap的操作也会很有用，因为插入和删除操作也涉及到bitMap的读和写，我们用封装好的位运算操作会让代码更加简洁和干净。

- 同时，在一个Page发生了修改之后，这个Page就变成了脏页，需要标记为dirty，并让BufferPool在适当的时机写回磁盘存储的文件中
- 之后所有涉及到页修改的操作都需要通过BufferPool对外提供的方法进行，这其中执行的逻辑是BufferPool先判断要操作的页在不在Buffer中，如果不在就先去磁盘中把对应的Page读进来，然后在这个页上进行相关操作



#### 运算符

在实现了存储系统中的修改操作之后，我们才能进一步实现Delete和Insert操作的运算符，这里的运算符在进行页的修改的时候都需要通过BufferPool提供的方法来修改，我们只需要再次基础上实现每个运算符的fetchNext方法就行。

- 值得注意的是，Insert和Delete操作的运算符返回的结果依然是元组的形式，但是这个元组只有1个属性，并且这个属性值代表了Insert和Delete操作影响的元组数量。

### 页的置换

在lab2的最后，我们还需要实现页的置换算法，因为BufferPool是有容量上限的，如果我们在不断读入页的时候把容量用完了，那么就必须将一部分页置换出去，这样才能让新的页不断读入内存中，常见的页置换策略有LRU、Clock等等，我们在lab2中也需要自己实现一种Buffer中页的置换算法。

于是我选择了最简单的实现方式——替换Buffer中的第一个页，具体方法是每当要发生置换的时候，就用Buffer的迭代器获取位于最前面的页，然后将其替换，这个实现过程非常简单，也比较有效。





