# CMU15-445-Project3



> CMU15-445的Project3，目标是实现若干种数据库操作的查询执行器，这一Project的重大意义在于理解Bustub数据库引擎中对于executor的模块和迭代查询处理模型的设计和实现。

## START

这一个Project中我们需要实现数据库中多种操作的执行器，包括：

- 访问方式：顺序扫描
- 数据修改：查询，删除，更新
- 复杂操作：循环嵌套Join，Hash Join，聚合，Limit和Distinct

同时Bustub并不支持真正的SQL语句，所以执行的都是手动编写的查询计划。同时Bustub中采用的是迭代模型，在这种模型中，每一个查询计划的执行器都需要实现一个`Next()`函数，当调用执行器的这个函数的时候，执行器会返回一个元组或者一个提示已经没有更多元组了的标志，通过这个方法，每个执行器实现一个循环结构来持续调用子结点的`Next()`函数，从子结点中一个接一个检索元组并进行操作。

在Bustub中，每个`Next()`函数会返回一个元组和代表了这个元组的RID(RID实际上就是元组的唯一标识)，下面我们就要按顺序来实现一堆不同操作的执行器，包括：

- `seq_scan_executor`
- `insert_executor`
- `update_executor`
- `delete_executor`
- `nested_loop_join_executor`
- `hash_join_executor`
- `aggregation_executor`
- `limit_executor`
- `distinct_executor`

每个执行器需要实现`Init(), Next()`这两个函数，其中`Init()`负责将该操作的执行器进行初始化，比如设置对应的要扫描的表，而`Next()`则用来完成单个元组的检索。

## Bustub中的系统架构设计

在开始写这部分的代码之前，我们需要先熟悉一下这部分相关的代码结构，并学习一下bustub的系统设计模式。

### Catalog数据结构设计

Bustub中的catalog定义了一系列用来维护数据元信息的数据结构，包括`Catalog, TableInfo, IndexInfo`等等，`TableInfo`和`IndexInfo`分别代表了一个表/索引的元信息，而Catalog定义了整个数据库系统的元信息，包括了若干表和索引组成的数组。而一个表和索引中最关键的信息就是一个被称为`Schema`的类，这个类中定义了**表/索引每一列的性质**，包括数据类型，长度限制等等，一个列所具有的特征用`Column`这个类来表示，Schema中就集成了一系列的Column并提供对外访问的API。

综上所属，Bustub的Catalog部分的关键类之间的组合关系是`Column --> Schema --> TableInfo/ IndexInfo --> Catalog` 

### Execution数据结构设计

#### 查询上下文的设计

查询上下文是一个用来**保存查询处理的时候的上下文环境**的泪，它的定义中包含了如下内容：

```cpp
class ExecutorContext {
 private:
  /** The transaction context associated with this executor context */
  Transaction *transaction_;
  /** The datbase catalog associated with this executor context */
  Catalog *catalog_;
  /** The buffer pool manager associated with this executor context */
  BufferPoolManager *bpm_;
  /** The transaction manager associated with this executor context */
  TransactionManager *txn_mgr_;
  /** The lock manager associated with this executor context */
  LockManager *lock_mgr_;
};
```

我们可以看到它包含了前面提到的`Catalog`类，还包括了管理Buffer的BufferManager和管理事务的TransactionManager，这些信息共同构成了查询计划执行过程中的上下文。

#### 查询执行器的设计

Bustub中抽象出了一个查询执行器类，定义为：`AbstractExecutor`，这个类中定义了一个执行器的基本结构，其定义的代码如下：

```cpp
class AbstractExecutor {
 public:
  /**
   * Construct a new AbstractExecutor instance.
   * @param exec_ctx the executor context that the executor runs with
   */
  explicit AbstractExecutor(ExecutorContext *exec_ctx) : exec_ctx_{exec_ctx} {}

  /** Virtual destructor. */
  virtual ~AbstractExecutor() = default;

  /**
   * Initialize the executor.
   * @warning This function must be called before Next() is called!
   */
  virtual void Init() = 0;

  /**
   * Yield the next tuple from this executor.
   * @param[out] tuple The next tuple produced by this executor
   * @param[out] rid The next tuple RID produced by this executor
   * @return `true` if a tuple was produced, `false` if there are no more tuples
   */
  virtual bool Next(Tuple *tuple, RID *rid) = 0;

  /** @return The schema of the tuples that this executor produces */
  virtual const Schema *GetOutputSchema() = 0;

 protected:
  /** The executor context in which the executor runs */
  ExecutorContext *exec_ctx_;
};
```

每个执行器对象会拥有自己的输出格式(`OutputSchema`)和执行上下文，这个类被不同SQL操作对应的执行器继承，形成一系列执行器。

Bustub中执行器的设计采用了**工厂模式**，定义了一个`ExecutorFactory`以及静态函数`CreateExecutor`来生成一个执行器。这个静态函数的定义如下所示：

```c++
static std::unique_ptr<AbstractExecutor> 
  CreateExecutor(ExecutorContext *exec_ctx, const AbstractPlanNode *plan);
```

它会根据执行上下文和执行计划中包含的操作类型，返回对应操作的执行器。

#### 查询计划的设计

查询计划在Bustub中被定义为了文件`abstract_plan.h` 中的一个抽象类`AbstractPlanNode`，这个抽象类中记录了查询计划节点**所应该具有的抽象特征和元信息**，其关键的定义如下：

```c++
class AbstractPlanNode {
 private:
  /**
   * The schema for the output of this plan node. In the volcano model, every plan node will spit out tuples,
   * and this tells you what schema this plan node's tuples will have.
   */
  const Schema *output_schema_;
  /** The children of this plan node. */
  std::vector<const AbstractPlanNode *> children_;
};
}  // namespace bustub

```

其中`Output_schema`表示输出结果(按照常识，查询的输出结果也应该是一个表的形式)的Schema，而children则是记录了当前节点的子结点的指针，因为一个查询计划是一个树形的结构，由一系列`AbstractPlanNode`组成，然后`AbstractPlanNode`被不同的SQL操作对应的查询计划继承，衍生出了如`seq_scan_plan`之类的一系列具体的查询计划类。

#### 查询执行引擎的设计

查询执行引擎的类`ExecutionEngine`对外提供了一个执行查询计划的函数`Execute`，它的定义如下：

```c++
bool Execute(const AbstractPlanNode *plan, std::vector<Tuple> *result_set, Transaction *txn,
               ExecutorContext *exec_ctx) {
    // Construct and executor for the plan
    auto executor = ExecutorFactory::CreateExecutor(exec_ctx, plan);

    // Prepare the root executor
    executor->Init();

    // Execute the query plan
    try {
      Tuple tuple;
      RID rid;
      while (executor->Next(&tuple, &rid)) {
        if (result_set != nullptr) {
          result_set->push_back(tuple);
        }
      }
    } catch (Exception &e) {
      // TODO(student): handle exceptions
      LOG_DEBUG("%s", e.what());
    }

    return true;
  }
```

这个函数会根据查询计划和上下文来创建对应的执行器，并用执行器来实现对应的数据库操作。



## 运算执行器的实现

### 顺序扫描

顺序扫描执行器 `SeqScanExecutor`是最基本的一个执行器，它会在一个表上进行迭代式的扫描并返回其元组的信息，每次返回一个，一个顺序扫描被一个`SeqScanPlanNode`来定义，这个执行计划结点会指定需要扫描的表，同时这个结点中可能会包含一个判断条件(Predicate)，需要过滤掉不符合要求的元组。

同时Bustub中已经提高了一个`TableIterator`对象可以用来遍历一个表，并且在表达式类`AbstractExpression`中提供了一个`Evaluate()`函数可以用来判断一个元组是否满足Predicate中的表达式，它会返回一个Value对象，将其转化成bool类型之后就可以表示元组是否满足表达式中的判断条件。

- 所以顺序扫描其实对应的就是有条件的`SELECT`操作

顺序扫描的代码实现如下：

```c++
SeqScanExecutor::SeqScanExecutor(ExecutorContext *exec_ctx, const SeqScanPlanNode *plan) 
: AbstractExecutor(exec_ctx), plan_(plan) {}

void SeqScanExecutor::Init() {
  table_info_ = exec_ctx_->GetCatalog()->GetTable(plan_->GetTableOid());
  table_iter = table_info_->table_->Begin(exec_ctx_->GetTransaction());
  table_end_iter = table_info_->table_->End();
}

bool SeqScanExecutor::Next(Tuple *tuple, RID *rid) {
  for (; table_iter != table_end_iter; table_iter ++) {
    exec_ctx_->GetLockManager()->LockShared(exec_ctx_->GetTransaction(), table_iter->GetRid());
    *tuple = *table_iter;
    *rid = tuple->GetRid();
    if (plan_->GetPredicate() != nullptr) {
      if (plan_->GetPredicate()->Evaluate(tuple, plan_->OutputSchema()).GetAs<bool>()) {
        table_iter ++;
        return true;
      }
    } else {
      table_iter ++;
      return true;
    }
  }
  return false;
}
```



### 插入

插入操作需要实现的功能是将元组插入表中，并且插入操作分成两种不同的类型：

- 第一种插入操作是指，需要插入的值本身包含在查询计划的结点中，我们称为行插入
- 第二种插入操作是指插入的值来自于一个子结点上的执行器，比如InsertNode下面可以有一个SeqScan的子结点

在执行器初始化的时候，我们需要先从Catalog中获取表的信息，然后再使用`TableHeap`来完成表的更新。所以插入操作的初始化实现如下(注意要在InsertExecutor中定义相关的变量)：

```cpp
void InsertExecutor::Init() {
  Catalog *catalog = GetExecutorContext()->GetCatalog();
  table_info_ = catalog->GetTable();
  std::string table = table_info_->name_;
  index_info_vector_ = catalog->GetTableIndexes(table);
  // 对应两种不同的查询方式
  if (plan_->IsRawInsert()) {
    iter_ = plan_->RawValues().begin();
  } else {
    // 如果插入的数据是从子结点来的，就对子结点进行初始化
    child_executor_->Init();
  }
}
```

此外我们把插入数据的过程单独抽象成一个函数`Insert`，并且在`Next`中调用，然后在`Next`中分成两种情况的insert分别写就好了：

```c++
bool InsertExecutor::Insert(Tuple &tuple, RID *rid) {
  TableHeap *table_heap = table_info_->table_.get();
  table_heap->InsertTuple(tuple, rid, GetExecutorContext()->GetTransaction());
  for (auto &index_info: index_info_vector_) {
    HASH_TABLE_INDEX_TYPE *hash_index = reinterpret_cast<HASH_TABLE_INDEX_TYPE*>(index_info->index_.get());
    IndexWriteRecord index_record{*rid, plan_->TableOid(), WType::INSERT,
                                  tuple, index_info->index_oid_, GetExecutorContext()->GetCatalog()};
    GetExecutorContext()->GetTransaction()->AppendTableWriteRecord(index_record);
    hash_index->InsertEntry(tuple.KeyFromTuple(table_info_->schema_, index_info->key_schema_,
                                               index_info->index_->GetMetadata()->GetKeyAttrs()),
                            *rid, GetExecutorContext()->GetTransaction());
  }
}

bool InsertExecutor::Next([[maybe_unused]] Tuple *tuple, RID *rid) {
  if (!plan_->IsRawInsert()) {
    if (child_executor_->Next(tuple, rid)) {
      Insert(tuple, rid);
      return true;
    }
    return false;
  }
  if (iter_ != plan_->RawValues().end()) {
    Tuple insert_tuple(*iter_ ++, &table_info_->schema_);
    Insert(insert_tuple, rid);
    return true;
  }
  return false;
}

```

### 更新操作

更新操作是在表中修改一个元素并在索引中进行更新，这个执行器的子执行器会提供需要修改的元组的RID信息，同时更新操作总是从子执行器(查询计划的子结点)中获得要操作的元组信息，即Update一定会跟一个Select操作(虽然SQL里面没有，但是数据库系统实际执行过程中就是要先Select出所有的元组然后再Update的)

同时Bustub中提供了`GenerateUpdatedTuple`来根据查询计划中的Schema生成修改后的元组，并在TableHeap里进行表的修改。

```c++
bool UpdateExecutor::Next([[maybe_unused]] Tuple *tuple, RID *rid) {
  if (!enable_logging) {
    if (exec_ctx_->GetTransaction()->IsSharedLocked(*rid)) {
      if (!exec_ctx_->GetLockManager()->LockUpgrade(exec_ctx_->GetTransaction(), *rid)) {
        return false;
      }
    } else {
      if (!exec_ctx_->GetLockManager()->LockExclusive(exec_ctx_->GetTransaction(), *rid)) {
        return false;
      }
    }
  }
  if (child_executor_->Next(tuple, rid)) {
    Tuple t = GenerateUpdatedTuple(*tuple);
    if (table_info_->table_->UpdateTuple(t, *rid, exec_ctx_->GetTransaction())) {
      table_info_->table_->GetTuple(*rid, tuple, exec_ctx_->GetTransaction());
      return true;
    } else {
      throw Exception("Update Error!");
    }
  }
  return false;
}
```



### 未完待续