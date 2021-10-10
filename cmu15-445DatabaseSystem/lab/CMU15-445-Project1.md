# CMU15-445-Project1

## Start Up

Project1算是正式开始了，CMU15-445的这门课的教学安排和学校里的同类型课程《数据库系统》差别较大，学校里的课程是先讲了SQL的语法，范式开始讲，这门课除了前面复习一下SQL的高级用法剩下的课程内容都在讲数据库系统的设计原理，从数据库存储开始讲，然后又讲了缓冲池的设计和索引。Project1主要包含的内容就是文件存储和缓冲池的实现，需要分别实现以下几个模块：

- LRU替换算法
- 缓冲区管理器的各项功能
- 并行的缓冲区管理器

同时已经提供的代码中封装好了页的数据结构`Page`，磁盘的**读写操作API**`Disk Manager`可以使用，Page类的关键属性和函数定义如下：

```c++
class Page {
  inline char *GetData() { return data_; }
  inline page_id_t GetPageId() { return page_id_; }
  inline bool IsDirty() { return is_dirty_; }
  char data_[PAGE_SIZE]{};
  page_id_t page_id_ = INVALID_PAGE_ID;
  int pin_count_ = 0;
  bool is_dirty_ = false;
  ReaderWriterLatch rwlatch_;
}
```

可以看到代码中已有的页类可以修改页中的数据，获得页号和是否为脏页等信息，同时可以修改`pin_count`，而

`Disk Manager` 可以管理一个数据库系统中页的分配和回收，并提供了在内存和磁盘之间对页进行读写的API，为数据库系统提供了一个逻辑意义上的文件层，它提供的关键函数有：

```c++
class DiskManager {
  void WritePage(page_id_t page_id, const char *page_data);
  void ReadPage(page_id_t page_id, char *page_data);
  void WriteLog(char *log_data, int size);
  bool ReadLog(char *log_data, int size, int offset);
  page_id_t AllocatePage();
  void DeallocatePage(page_id_t page_id);
  int GetNumFlushes() const;
  bool GetFlushState() const;
  int GetNumWrites() const;
  inline void SetFlushLogFuture(std::future<void> *f) { flush_log_f_ = f; }
  inline bool HasFlushLogFuture() { return flush_log_f_ != nullptr; }
}
```

后面缓冲区管理器的实现非常依赖这些已有的读写API，事实上基本都是对DiskManager中函数的一些调用和包装。

此外，已有的代码中对一些常见的数据类型进行了包装，比如表示页号/物理帧号的`page_id_t， frame_id_t` 

## Step1-LRU读写策略

### 总体目标

首先我们先来实现一个LRU的页管理和替换策略，Bustub中定义了一个类`LRUReplacer`继承了`Replacer`专门来实现LRU的替换算法，并在`BufferManager`中进行调用，其实质是维护了一个由`page_id`组合成的双向链表，然后实现和页/页表管理有关各种操作，同时`BufferManager`中还有物理块和页表，这里的`LRUReplacer`只研究页号，具体的块的置换操作要在`BufferManager`中进行，这些操作包括：

- Victim：从Replacer中移除最近最早访问的页号，即双向链表的最后面一个位置
- Pin：当一个页被pin的时候调用，这表明这个数据帧已经被某个进程调用了，不能参与LRU的替换，必须要保留，因此需要将这个页从`Replacer`中移除
- Unpin：当一个页的pin_count变成0的时候调用，这个时候这个页已经没有进程使用了，需要将这个页加入双向链表的最前面，即最近刚访问过
- Size：获得参与LRU的页的数量

因此`LRUReplacer`中我使用这样几个自定义的属性：

```c++
int num_pages;
std::list<frame_id_t> frames;
std::unordered_map<frame_id_t, std::list<frame_id_t>::iterator> mp;
std::mutex mutex;
```

两个关键的STL分别表示用于LRU替换的双向链表和一个记录帧号-链表迭代器的map用于快速获取元素，而mutex是一个互斥锁，用于保证并发访问时的一致性

### Victim的实现

Victim就是替换掉双向链表中处于最后的块，然后将这个frame_id记录下来就可以：

```c++
bool LRUReplacer::Victim(frame_id_t *frame_id) {
  std::lock_guard<std::mutex> lock(mutex);
  if (frames.empty()) {
    return false;
  }
  *frame_id = frames.back();
  frames.pop_back();
  mp.erase(*frame_id);
  return true;

```

### Pin的实现

Pin就是将某个frame_id暂时从双向链表中踢出，供进程访问和使用：

```c++
void LRUReplacer::Pin(frame_id_t frame_id) {
  std::lock_guard<std::mutex> lock(mutex);
  // find the index of the frame
  std::unordered_map<frame_id_t, std::list<frame_id_t>::iterator>::iterator mp_iter = mp.find(frame_id);
  if (mp_iter != mp.end()) {
    // clear the existing frame information
    std::list<frame_id_t>::iterator frame_iter = mp_iter->second;
    frames.erase(frame_iter);
    mp.erase(mp_iter);
  }
}
```

### Unpin的实现

Unpin就是将一个块加入LRU双向链表中，因为是最近刚访问的所以要放在链表的头部：

```c++
void LRUReplacer::Unpin(frame_id_t frame_id) {
  std::lock_guard<std::mutex> lock(mutex);
  std::unordered_map<frame_id_t, std::list<frame_id_t>::iterator>::iterator mp_iter = mp.find(frame_id);
  if (mp_iter == mp.end()) {
    frames.push_front(frame_id);
    mp[frame_id] = frames.begin();
  }
}
```



## Step2-缓冲区管理器

下一步是要实现缓冲区的管理，Bustub的starter code中已经给出了`BufferManager,BufferManagerInstance`的定义其中`BufferManager`是一个抽象类，我们需要实现的是`BufferManagerInstance` ，这个类需要借助`DiskManager`和`LRUReplacer`完成对缓冲区的管理，其中`DiskManager`负责实际的磁盘块读写，而LRUReplacer可以进行LRU规则的块替换，此外我们还可以使用如下几个数据结构：

```c++
	/** Array of buffer pool pages. */
  Page *pages_;
  /** Page table for keeping track of buffer pool pages. */
  std::unordered_map<page_id_t, frame_id_t> page_table_;
  /** List of free pages. */
  std::list<frame_id_t> free_list_;
```

- `pages_`是存放了各个page的物理帧
- `page_table_`是页表，用于将页号转换成物理帧号
- `free_list_`是一个双向链表，用于存放空的物理帧

在实现具体的功能的时候就需要结合这些数据结构一起考虑，缓冲区管理器需要实现的功能包括：

- `FetchPgImp(page_id)` 获得一个页的使用权，如果这个页在缓冲区中不存在，那么就从磁盘中读取新的页到缓冲区，这中间可能会发生页的替换
- `UnpinPgImp(page_id, is_dirty)` 将某个页unpin，即取消对该页的引用
- `FlushPgImp(page_id)` 将一个页的信息写回磁盘中
- `NewPgImp(page_id)` 在磁盘上新增一个页
- `DeletePgImp(page_id)` 删除某个页
- `FlushAllPagesImpl()` 将所有页的信息都写回磁盘中

同时所有的操作都要加锁保证一致性，具体的操作其实就是在每个函数最前面写上：

```c++
std::lock_guard<std::mutex> lock(latch_);
```

虽然看起来要写的东西很多，但是这些任务都有一个基本的pipeline，首先需要根据页号去查询对应的帧号，然后从缓冲区拿到对应的页进行某些操作，比如修改页的某些元数据，调用DiskManager和LRUReplacer，修改页表等等，以Fetch操作为例，我们需要先判断要找的页在不在缓冲区，如果在就直接把对应的引用返回就可以，如果不在就需要从磁盘去读取这个页(调用DiskManager)，如果缓冲区没有满就拿出一个空的帧来存放页，如果满了就需要调用LRU置换来换走某个帧上的页，做完之后我感觉这个功能是第二部分最复杂的，综合性也是最强的，其具体的实现代码如下：

```c++
Page *BufferPoolManagerInstance::FetchPgImp(page_id_t page_id) {
  // 1.     Search the page table for the requested page (P).
  // 1.1    If P exists, pin it and return it immediately.
  // 1.2    If P does not exist, find a replacement page (R) from either the free list or the replacer.
  //        Note that pages are always found from the free list first.
  // 2.     If R is dirty, write it back to the disk.
  // 3.     Delete R from the page table and insert P.
  // 4.     Update P's metadata, read in the page content from disk, and then return a pointer to P.
  std::lock_guard<std::mutex> lock(latch_);
  auto page = page_table_.find(page_id);
  frame_id_t frame_id = INVALID_PAGE_ID;
  // find the page and pin it
  if (page != page_table_.end()) {
    frame_id = page_table_[page_id];
    pages_[frame_id].pin_count_ ++;
    replacer_->Pin(frame_id);
    return &pages_[frame_id];
  }
  // not find the page and find a replacement
  if (!free_list_.empty()) {
    // the free list is not empty
    frame_id = free_list_.front();
    free_list_.pop_front();
  } else {
    // victim the page
    if (replacer_->Victim(&frame_id)) {
      // write back the dirty page
      if (pages_[frame_id].IsDirty()) {
        pages_[frame_id].is_dirty_ = false;
        disk_manager_->WritePage(pages_[frame_id].GetPageId(), pages_[frame_id].GetData());
      }
      // clear the page in the page table
      page_table_.erase(pages_[frame_id].GetPageId());
    } else {
      return nullptr;
    }
    pages_[frame_id].ResetMemory();
    disk_manager_->ReadPage(page_id, pages_[frame_id].GetData());
    pages_[frame_id].page_id_ = page_id;
    pages_[frame_id].pin_count_ ++;
    page_table_[page_id] = frame_id;
    return &pages_[frame_id];
  }
  return nullptr;
}
```

- 其他的几个功能也都是类似的写法，不过没有Fetch那么复杂

## Step3-Parallel Buffer Pool Manager

第三步是实现具有**多个Buffer Pool的Manager**以支持多线程的访问，这一部分的主要任务就是创建多个缓冲区管理器的实例用来并发访问，并通过一定的规则建立起页和缓冲池的映射关系，然后在缓冲池中进行对应页的操作。需要实现的功能包括：

- `ParallelBufferPoolManager(num_instances, pool_size, disk_manager, log_manager)`
- `~ParallelBufferPoolManager()`
- `GetPoolSize()`
- `GetBufferPoolManager(page_id)`
- `FetchPgImp(page_id)`
- `UnpinPgImp(page_id, is_dirty)`
- `FlushPgImp(page_id)`
- `NewPgImp(page_id)`
- `DeletePgImp(page_id)`
- `FlushAllPagesImpl()`

