# CMU-15/440 Distributed Systems 3: Golang and Synchronization 

> 分布式系统课程15-440学习笔记3，这一节主要讲go语言和go语言中的并发与同步模型

## 并发中的一些关键概念

要解释清楚什么是并发，首先应该要知道这样几个基本的概念：

- 临界区Critical Section，指的是访问被共享的那部分资源的代码片段
- 竞争条件Race Condition，多个线程同时访问代码中的临界区并对共享的资源进行更新，导致无法预知的后果
- 非确定性程序Indeterminate Program，一个或者多个竞争条件，导致程序最后的输出不是确定的
- 互斥Mutual Exclusion，即保证每个时刻都只有一个线程/进程可以访问临界区，以此来避免竞争，同时又要保证线程/进程只需等待有限的时间，即不浪费资源，并保证访问的公平性

## Golang的并发模型

Golang(即Go语言)设计了channel和go routines两种语法特性来实现它的线程模型。其中，channel的作用是用来传递信息并实现go routine的同步，同时提供回调的服务。而go routine则是由Go创建的可以独立执行的函数，并且有独立的调用栈，可以看作是一种轻量级的线程。Go语言的设计理念是，**用通信来实现内存的共享，而不是用共享内存来实现进程间的通信**。

channel是一个有长度限制的FIFO队列，一边负责进(Insert)，一边负责出(Removal)，如果一个channel的容量是0，那么它可以作为一个sync point使用，即Insert了1个之后就不能再Insert了，需要等别的进程Remove了才能继续Insert，而第二次Insert对应的语句会进入block状态，直到Remove之后，这就起到了晋城之间通信的作用。而一个容量不为0的channel则可以容纳对应数量的特定类型的数据。下面是一个用channel包装成互斥锁的例子：

```go
type Mutex struct {
     mc chan int
}
// Create an unlocked mutex 
func NewMutex() *Mutex {
	m := &Mutex{make(chan int, 1)}
	m.Unlock() # Initially, channel empty == locked 	return m
}
// Lock operation, take a value from the channel 
func (m *Mutex) Lock() {
	<- m.mc # Don't care about value }
	func (m *Mutex) Unlock() {
	m.mc <- 1 # Stick in value 1. 
}
```

但是Go的channel也有一些问题，比如channel在初始化的时候就需要指定容量的大小，不能定义一个无限大的channel，此外，channel没有办法判断它是不是空的，当我们从channel中读取数据之后，就不能把数据再放回channel的头部，这就是说我们没法对channel的数据做校验，并且也无法