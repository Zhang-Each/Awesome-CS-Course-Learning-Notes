# CMU15-445-Project2

> 上一个Project中，我们实现了Bustub中的Buffer Pool模块，实现了包括LRU置换算法，Buffer Manager和并行Buffer等内容，而这次的Project2需要我们自己实现Bustub中的Index模块

## START

Bustub中的索引采用的是Hash Index，并且今年的Project中采用了可扩展的**Extendable Hash**作为索引的核心组成部分，并要求实现并发的控制等操作。

整个实验分成了三个部分，分别是Page相关的数据结构的实现，Hash表本身的实现和并发控制的实现，下面进入正题。

## PAGE LAYOUTS

第一部分我们需要实现索引页的基本数据结构，前面我们已经提到过，Buffer Manager的功能是对页进行管理，里面集成了Disk Manager，LRU等多个组件，就是为了对各种各样的页的读写和置换进行有效的管理，而现在就轮到了我们真的实现其中的一些页的时候。可扩展Hash中，页分成了两种，一种是目录页(Directory page)，另一种是桶页面(Bucket Page)，目录页面主要保存的是**Hash的结果和桶的对应关系**(后面会详细说可扩展Hash在这方面的设计)，而桶主要用来保存key-value的数据对。

同时我们不能使用页以外的东西存储Hash表的信息，也就是说需要做到自包含，即可以只通过读入Hash表来复原整个可扩展Hash索引。

### Directory Page

首先是第一部分目录页的设计，目录页用来保存hash表中的元数据，并且被分成了以下几个部分：

| 变量名             | 大小       | 描述                                               |
| ------------------ | ---------- | -------------------------------------------------- |
| `page_id_`         | 4 bytes    | 自身的页编号                                       |
| `lsn_`             | 4 bytes    | 日志序列号(这个Project中暂时不用)                  |
| `global_depth_`    | 4 bytes    | 可扩展Hash的全局深度                               |
| `local_depths_`    | 512 bytes  | 可扩展Hash的局部深度，是一个数组，每个桶对应一个值 |
| `bucket_page_ids_` | 2048 bytes | 桶和Page_id 的对应关系                             |

我们不难发现其实这个设计非常合理，而我们需要实现的函数主要都是一些对这个基本数据结构的查询和修改，包括：

```c++
uint32_t HashTableDirectoryPage::GetGlobalDepth() {
  return global_depth_;
}

uint32_t HashTableDirectoryPage::GetGlobalDepthMask() {
  uint32_t mask = pow(2, GetGlobalDepth()) - 1;
  return mask;
}

page_id_t HashTableDirectoryPage::GetBucketPageId(uint32_t bucket_idx) {
  page_id_t page_id = bucket_page_ids_[bucket_idx];
  return page_id;
}

void HashTableDirectoryPage::SetBucketPageId(uint32_t bucket_idx, page_id_t bucket_page_id) {
  bucket_page_ids_[bucket_idx] = bucket_page_id;
}

uint32_t HashTableDirectoryPage::GetLocalDepth(uint32_t bucket_idx) {
  page_id_t page_id = GetBucketPageId(bucket_idx);
  return local_depths_[page_id];
}

void HashTableDirectoryPage::SetLocalDepth(uint32_t bucket_idx, uint8_t local_depth) {
  page_id_t page_id = GetBucketPageId(bucket_idx);
  local_depths_[page_id] = local_depth;
}

void HashTableDirectoryPage::IncrLocalDepth(uint32_t bucket_idx) {
  page_id_t page_id = GetBucketPageId(bucket_idx);
  local_depths_[page_id] ++;
}

void HashTableDirectoryPage::DecrLocalDepth(uint32_t bucket_idx) {
  page_id_t page_id = GetBucketPageId(bucket_idx);
  local_depths_[page_id] --;
}
```

这里的绝大部分内容都没什么难度，比较有点意思的只有`GetGlobalDepthMask`，这个方法可以获得一个全局深度的mask用来处理key经过hash函数后得到的值。这是因为可扩展哈希中，我们需要将一个key映射到目录的索引中，这个过程使用的规则就是：
$$
\mathrm{DirectoryIndex}=\mathrm{Hash}(key)\quad\&\quad \mathrm{Mask}
$$
这里的Mask是一个32位无符号整数，并且其二进制表示中有和全局深度相同个数的1，比如全局深度是3那么Mask就是二进制的111也就是7，所以返回的结果是`0x00000007` 



### Bucket Page

Bucket Page是用来存放具体key-value对的页，它的核心数据结构是三个数组：

- `occupied_` : 数组中的第i个bit代表array中的第i个位置是否已经被占用
- `readable_` : 数组中的第i个bit代表array中的第i个元素是否可读
- `array_` : 用来存储具体的key-value对

因此初始代码中这样定义了三个数组：

```c++
template <typename KeyType, typename ValueType, typename KeyComparator>
class HashTableBucketPage {
 private:
  //  For more on BUCKET_ARRAY_SIZE see storage/page/hash_table_page_defs.h
  // 一个char代表一个byte，包含了8个bit，而操作的时候要精确到bit
  char occupied_[(BUCKET_ARRAY_SIZE - 1) / 8 + 1];
  // 0 if tombstone/brand new (never occupied), 1 otherwise.
  char readable_[(BUCKET_ARRAY_SIZE - 1) / 8 + 1];
  MappingType array_[0];
};
```

值得注意的是这里的`occupied, readable`被定义成了char数组，而1个char的大小是1byte，也就是8bit，我们之前说两个数组用第i位表示key-value数组的第i个位置是否被占用/可读，而这几个标记数组中每8个位置就凑成一个char，所以我们在查询某个位置是否被占用/可读的时候，要先将其映射到对应的char中，然后**通过位运算对8个bit中指定的bit进行操作**。

事实上写代码的时候我真的用到了位运算的操作，包括and, or和shift等等，作业的要求是我们要实现这些内容：

- `Insert` 插入新的Key-value对
- `Remove` 移除旧的Key-value对
- `IsOccupied` 判断某个位置是否被占用
- `IsReadable` 判断某个位置是否可读
- `KeyAt` 找到某个位置的Key
- `ValueAt `找到某个位置的Value

两个判断函数的实现方式如出一辙，就是将输入的index转化到对应的char和对应的位置上，然后通过位运算取出对应位置的数据进行操作，比如判断被占用，只要把那个位置上的bit拿出来，和1求一个and就可以判断那个位置是不是1，值得注意的是，虽然按道理说bit应该是从左往右数的，但是因为二进制表示是从右往左排列的，为了操作方便，我们在每个char里，bit和索引的对应关系就按照从右往左的方式排序了，即**从右往左，8个bit依次是当前char的0-7位**，输入的索引号先对应到某个具体的char上(`index / 8`)，然后用余数代表在一个char中的偏移量，也就是排位。不过好消息是这么做区别不大，甚至更方便了。

```c++
template <typename KeyType, typename ValueType, typename KeyComparator>
bool HASH_TABLE_BUCKET_TYPE::IsOccupied(uint32_t bucket_idx) const {
  uint32_t byte_idx = bucket_idx / 8;
  uint32_t idx = bucket_idx % 8;
  // 通过位运算，找到byte_idx这个char(8-bit)的从右往左数第idx个元素，并判断是不是1
  return (occupied_[byte_idx] >> idx) & 1;
  //return (occupied_[bucket_idx / 8] >> (bucket_idx % 8)) & 1;
}
```

而`KeyAt`和`ValueAt`两个函数也是如出一辙，实现比较简单，就是找到对应位置上的pair然后返回key或者value即可：

```c++
template <typename KeyType, typename ValueType, typename KeyComparator>
KeyType HASH_TABLE_BUCKET_TYPE::KeyAt(uint32_t bucket_idx) const {
  return array_[bucket_idx].first;
}

template <typename KeyType, typename ValueType, typename KeyComparator>
ValueType HASH_TABLE_BUCKET_TYPE::ValueAt(uint32_t bucket_idx) const {
  return array_[bucket_idx].second;
}

```

> 这里要注意array是一个弹性数组，这是一个我没听说过的C++特性，但总之就是定义了`array_[0]`之后就可以把array当成数组来使用

比较有挑战性的是Insert和Remove两个函数，倒不是因为他们的操作很复杂，而是因为作业给我们设定好的函数签名比较诡异，比如Remove，作业给定的函数签名中并没有对应的index，而是给了key，value和一个comparator比较器，需要我们自己遍历整个bucket来找到要删除的位置，Insert也是类似，没有指定要我们插入到哪个位置，而是给了key和value让我们自己决定，这样一来也需要我们遍历整个bucket然后找到一个合适的位置来插入新的key-value对。

- 在Insert中，我们实际上就是要先找到一个没有被占用的空位
- 在Remove中，我们实际上要先找到待删除的Key和Value对应的位置，然后把readble和occupied两个数组对应位置设置成0

所以这两个函数最后实现的代码如下：

```c++
template <typename KeyType, typename ValueType, typename KeyComparator>
bool HASH_TABLE_BUCKET_TYPE::Insert(KeyType key, ValueType value, KeyComparator cmp) {
  for (uint32_t i = 0; i < BUCKET_ARRAY_SIZE; i ++) {
    if (!IsOccupied(i) && !IsReadable(i)) {
      SetReadable(i);
      SetOccupied(i);
      array_[i] = std::make_pair(key, value);
      //LOG_DEBUG("Occupied = %d", IsOccupied(i));
      //LOG_DEBUG("i = %d", i);
      return true;
    }
  }
  return false;
}

template <typename KeyType, typename ValueType, typename KeyComparator>
bool HASH_TABLE_BUCKET_TYPE::Remove(KeyType key, ValueType value, KeyComparator cmp) {
  for (uint32_t i = 0; i < BUCKET_ARRAY_SIZE; i ++) {
    if (IsReadable(i) && IsOccupied(i)) {
      if (cmp(key, KeyAt(i)) == 0 && value == ValueAt(i)) {
        occupied_[i / 8] &= ~(1 << i % 8);
        readable_[i / 8] &= ~(1 << i % 8);
        return true;
      }
    }
  }
  return false;
}
```



## HASH TABLE IMPLEMENTATION



## CONCURRENCY CONTROL