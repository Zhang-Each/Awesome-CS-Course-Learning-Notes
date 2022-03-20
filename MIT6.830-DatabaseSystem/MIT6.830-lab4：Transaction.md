# MIT6.830-lab4：Transaction

> MIT6.830 Database Systems课程Project的第四部分，主要实现一些事务处理和并发相关的功能，我们需要为SimpleDB设计一个二阶段锁的管理器

## Overview

在开始这个lab的coding之前，我们先来复习一些数据库中的基本常识。事务是数据库中一组原子操作的集合，原子性指的是事物中的一系列操作只能被完全执行，或者完全没有执行，不能出现只执行了其中部分操作的情况。数据库中的事务具有ACID四个特点：

- 原子性：就是上面说的，一个事务只能被完整的执行或者不执行
- 一致性：在事务开始之前和事务结束以后，数据库的完整性没有被破坏
- 独立性：数据库允许并发，但是多个事务之间不能互相影响，不能因为并发事务的交叉执行导致数据库的不一致
- 持久性：事务处理结束后，对数据的修改就是永久的，即便系统故障也不会丢失

而为了实现事务的这些特性，数据库往往采用被称为“锁”的机制来实现。锁代表了事务对数据表的访问权限，经典的共享锁和排他锁基本使用规则如下：

- 如果一个事务要对一个对象进行读操作，就必须持有它的共享锁
- 如果一个事务要对一个对象进行写操作，就必须持有它的互斥锁
- 多个事务可以同时持有一个对象的共享锁，而只有一个事务可以持有一个对象的互斥锁
- 如果只有一个事务t拥有一个对象的共享锁o，那么t可以自动将锁升级成一个排他锁

而这里所说的“对象”有不同的级别(也可以叫做粒度)，可以是一个数据表，可以是一个页，也可以是一个元组，在SimpleDB中，我们只考虑页级别的锁。

同时在申请锁的时候，如果一个事务需要的锁不能立即获得，那么它就需要进入阻塞状态，直到被分配到所需要的锁，数据库系统中一般会有一个专门管理锁的分配和释放的组件，我们可以称之为LockManager，下面我们正式开始实现一个自己的锁管理器以及SimpleDB中相关的东西。



## 代码实现

### LockManager

lab4的第一个任务就是在Buffer中实现加锁的操作，不过这需要我们先简单设计一个LockManager的雏形，并且在BufferPool中调用LockManager的方法来实现一些东西。LockManager的基本设计如下：

```Java
public class LockManager {
    private ConcurrentHashMap<TransactionId, Set<Lock>> transactionLockMap;

    private ConcurrentHashMap<PageId, Lock> pageLockMap;

    private ConcurrentHashMap<PageId, Permissions> pagePermMap;

    public LockManager()

    public void lock(TransactionId transactionId, PageId pageId, Permissions permission)

    public void unlock(TransactionId transactionId, PageId pageId)

    public boolean isLocked(TransactionId transactionId, PageId pageId)

}

```

实际上LockManager只需要提供几个能满足并发访问需求的Hash表就可以了，因为我们设计的是页级别的锁，所以这些Hash建立的也就是PageId和Lock之间的对应关系，这里的Lock是我们设计的一个锁的类，在这一步的时候，Lock还不需要非常具体的实现，我们只要先新建一个类就行。

实现了LockManager之后，我们需要在原有的BufferPool代码基础上进行改进，比如`getPage()`这个方法中，就需要在最前面加上一行代码：

```Java
this.lockManager.lock(tid, pid, perm);
```

来表示系统去读一个页之前，必须获得这个页的锁，而另外两个要实现的方法`unsafeReleasePage()`和`holdsLock()`则是对LockManager的一层封装。

### 锁的生命周期

下面我们要具体实现Lock类的内容，SimpleDB要求我们实现一个二阶段的锁。我们之前实现的插入操作也好，删除操作也好，还是其他需要读数据的方法，实际上都是需要经过`BufferPool.getPage()`这个方法来读取页(因为都需要用Database获取页数据，而Database类下管理页数据的就只能是BufferPool)，而这个方法我们在上一节已经为其添加了获取锁的过程，所以整个数据库中涉及到读写操作的都会经历申请锁这个环节。

我们下面来具体设计一个二阶段的锁(也就是Lock类)，该类的基本设计如下：

```java
public class Lock {
    private int readCount, writeCount;

    public void Lock()
    public synchronized void lock(LockManager lockManager, TransactionId tId, PageId pId, Permissions perm)
            throws InterruptedException, TransactionAbortedException {
        // 如果当前有读锁且想获得写锁，则进行锁升级
        if (isLockUpgrade(lockManager, tId, pId, perm)) {
            lockUpgrade(lockManager, tId, pId, perm);
        } else if (isLockRepeat(lockManager, tId, pId, perm)) {
            // 如果重复获得同一个锁/低级锁，则跳过
            return;
        } else if (perm.equals(Permissions.READ_ONLY)) {
            readLock(lockManager, tId, pId, perm);
        } else {
            writeLock(lockManager, tId, pId, perm);
        }
    }
    public boolean isLockUpgrade(LockManager lockManager, TransactionId tId, PageId pId, Permissions perm)
    public boolean isLockRepeat(LockManager lockManager, TransactionId tId, PageId pId, Permissions perm)
    public synchronized void writeLock(LockManager lockManager, TransactionId tId, PageId pId, Permissions perm)
            throws InterruptedException, TransactionAbortedException
    public synchronized void readLock(LockManager lockManager, TransactionId tId, PageId pId, Permissions perm)
            throws InterruptedException, TransactionAbortedException
    public synchronized void readUnlock()

    public synchronized void writeUnlock()

    public int getReadLockNum()

    public synchronized void lockUpgrade(LockManager lockManager, TransactionId tId, PageId pId, Permissions perm)
            throws InterruptedException, TransactionAbortedException {
        // 锁升级 释放读锁, 获得写锁并且更新页的读写类型
        readUnlock();
        writeLock(lockManager, tId, pId, perm);
    }
}

```

这里的关键实际上是设计一个锁升级的过程，在二阶段锁协议里，如果当前有读锁且想获得写锁，就会直接将读锁升级成写锁，而不是先释放读锁再申请写锁。

### Transaction

SimpleDB中，每个查询开始的时候都会创建一个TransactionId表示一个事务开始被执行，而一个事务结束的标志是调用了BufferPool中的`transactionComplete`方法，这个方法会让一个事务commit或者abort，我们下一步就是来实现这个方法。

```java
public void transactionComplete(TransactionId tid, boolean commit) {

        try {
            if (commit) {
                this.flushPages(tid);
            } else {
                this.restorePages(tid);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        Iterator<Map.Entry<Integer, Page>> iter = this.buffer.entrySet().iterator();
        while (iter.hasNext()) {
            Page page = iter.next().getValue();
            this.unsafeReleasePage(tid, page.getId());
        }
    }
```

事实上这个方法的实现逻辑很简单，如果一个事务commit了，那么就要保存对应的内容，将修改过的页写回磁盘中持久化保存，否则为了避免页被修改，就要重新读取这些页，之后我们需要将这些页的锁都进行释放。
