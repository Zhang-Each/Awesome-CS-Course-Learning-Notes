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

## 时间同步

当每台机器都有自己的时钟的时候，就可能会出现这样的情况，机器A上先发生的事情所记录的时间可能就会比机器B上后发生的事情所记录的时间要迟。也就是说，计算机之间的时钟不一定是完全一致的，可能有快有慢(这也叫做时钟漂移)，这就导致我们在一个分布式系统中，面对大量计算机，需要做好时间上的同步。

理想情况下的网络通信如下图所示，sender和receiver之间的时间完全同步，信息发送的时间差=传输所需的时间差。

![image-20220806142911782](https://raw.githubusercontent.com/Zhang-Each/Image-Bed/main/img/image-20220806142911782.png)

但是真实的网络往往是不同步并且不可信的。

### Cristian算法

这是一种非常简单的时间同步算法，如下图所示，当一台机器上的进程p要获得当前时间的时候，它会向时间服务器S发送一个消息，服务器S收到消息之后会马上回复，并在消息里写上当前的时间t，p在收到这个消息之后，将当前时间设置为t+RTT/2，其中RTT是完成整个消息收发过程所用的时间。

这个方法实际上将消息的来回传递看成是一个匀速的过程，所以S发出时间t和p收到结果之间的时间差就被认为是RTT/2，这个做法实际上依赖于

![image-20220806154829879](https://raw.githubusercontent.com/Zhang-Each/Image-Bed/main/img/image-20220806154829879.png)

### NTP协议(Network Time Protocol)

NTP协议是一个用来计算网络时间的标准化协议，用来**实现服务端和客户端之间的时间同步**(在NTP模型里，服务端是可以实时获取标准UTC时间的，也就是说服务端本身的时间是正确的时间)它的大致模型如下：

![image-20220806160603368](https://raw.githubusercontent.com/Zhang-Each/Image-Bed/main/img/image-20220806160603368.png)

在这个传输过程中，RTT可以表示为`t3-t0+t2-t1`，而客户端在t3这里的具体时间就会变成t3+offset，这里的offset就代表了时间的偏移量，它的计算方式为：
$$
offset=t2+RTT/2-t3=(t1-t0+t2-t3)/2
$$
这样一来，客户端就实现了和服务端的时间同步。

### 伯克利算法

伯克利算法(Berkeley Algorithm)是一个用来实现服务器组内时间同步的算法。它适用的场景是服务器不能获取标准的UTC时间，但是多台机器之间需要进行时间同步的场景。

在伯克利算法中，会有一个**Time Daemon**，它会向其他的机器进行轮询(poll)，并获取每台机器上的时间，然后再发送消息告诉每台机器要如何调整自己的时间。

![image-20220806163326208](https://raw.githubusercontent.com/Zhang-Each/Image-Bed/main/img/image-20220806163326208.png)

在这个过程中，Time Daemon会使用Christian这样的算法来估算二者之间的时间差。

