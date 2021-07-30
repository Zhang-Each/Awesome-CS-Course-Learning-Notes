# Lecture2：RPC&Thread

> 这一节笔记包含了MIT6.824分布式系统的Lecture2，主要讲了一些和go语言相关的知识以及线程，RPC等基础知识

- 本课程的lab选用go作为编程语言，因为go语言对多线程，RPC的支持较好，本节课主要讲的就是RPC，多线程和go中对应的语法特性

## 为什么选用go作为编程语言

- 对线程的支持较好(go-routine)
- 方便的RPC(远程过程调用)
- 类型安全，自带垃圾回收机制
- 线程安全+自带垃圾回收机制的特点使得go特别有竞争力
- 相对而言语法比较简单，容易学
- 官方文档中可参考的文档和教程比较翔实

## 线程Thread

​	  线程是一种非常好用的结构工具，但是也充满了技巧性，go语言中实现了go-routines可以非常方便地创建线程。线程允许一个程序立刻开始做很多事情，每个线程轮流执行，一个进程中的多个线程可以共享内存，并且每个线程单独持有PC，寄存器和执行栈。

### 为什么需要线程

​	  线程体现了一种并发性，在分布式系统中非常有用，比如客户端会给很多服务器同时发送请求并等待回复，服务器需要处理来自多个客户端的请求并且每个请求都可能引起阻塞，而且线程支持多核运行，可以提高程序的性能。

### 线程的替代品与线程存在的问题

​	  采用事件驱动的方式编程可以替代线程，用一个表来记录所有时间的状态信息，然后通过一个循环来不断地轮询每一个时间并进行处理，更新其状态，就可以代替多线程的代码，但是事件驱动的编程方法不能充分利用多个CPU，并且编程非常麻烦。但是使用多线程编程也存在很多问题，包括：

- 数据的共享，可能带中临界区的竞争等问题，可以使用锁解决
- 线程之间的交互和通信比较麻烦，但是go中做了比较好的封装，可以使用channel和go-routine等语法特性很方便解决。
- 线程死锁的问题

### 实例：网络爬虫

​	  网络爬虫是一种抓去web页面并对其进行解析的手段，其挑战在于并发量的提升和I/O的性能瓶颈，我们需要在短时间内爬取大量网页并且每个URL最好只能访问一次(为了不浪费带宽)，而常见的网络爬虫解决方案有两种：

- 串行爬虫Serial Crawler，对网页进行深度优先搜索，通过一个串行的调用序列不断爬取网页，但是一次只能爬一个网页
- 并发锁爬虫，为每个网页的爬取创建一个单独的线程，线程之间共享一个map用于记录哪些网页被爬了，但是这个map需要上锁来保证一个网页不会被爬两次，这样可以带来很高的并发量，保证线程安全

课程网站提供的材料中给出了一个爬虫的详细代码如下，go中往往使用channel作为多线程之间通信的方式，并用于暂时存储数据，并起到同步的作用

```go
func Serial(url string, fetcher Fetcher, fetched map[string]bool) {
	if fetched[url] {
		return
	}
	fetched[url] = true
	urls, err := fetcher.Fetch(url)
	if err != nil {
		return
	}
	for _, u := range urls {
		Serial(u, fetcher, fetched)
	}
	return
}

type fetchState struct {
	mu      sync.Mutex
	fetched map[string]bool
}

func ConcurrentMutex(url string, fetcher Fetcher, f *fetchState) {
	f.mu.Lock()
	already := f.fetched[url]
	f.fetched[url] = true
	f.mu.Unlock()

	if already {
		return
	}

	urls, err := fetcher.Fetch(url)
	if err != nil {
		return
	}
	var done sync.WaitGroup
	for _, u := range urls {
		done.Add(1)
		//u2 := u
		//go func() {
		// defer done.Done()
		// ConcurrentMutex(u2, fetcher, f)
		//}()
		go func(u string) {
			defer done.Done()
			ConcurrentMutex(u, fetcher, f)
		}(u)
	}
	done.Wait()
	return
}

func ConcurrentChannel(url string, fetcher Fetcher) {
	ch := make(chan []string)
	go func() {
		ch <- []string{url}
	}()
	coordinator(ch, fetcher)
}
```

- ConcurrentChannel中Master创建一个worker并用来爬取网页，worker通过channel发送一个页面中的URL切片，并且多个worker共享一个channel，Master从中逐渐读取URL的信息并记录在map上面



## 远程过程调用RPC

​	  远程过程调用是实现分布式系统各种机制的最关键部分之一，本课程中所有的lab都要使用RPC，RPC的目的是提供一种易于编写的客户端/服务端交互机制，同时隐藏网络通信协议的细节，提供灵活的数据格式转换。

![image-20210708170121218](static/image-20210708170121218.png)

### RPC的go代码实现

本课程提供了一个key-value存储服务器的go语言代码实例，通过go的RPC库来实现，具体的代码如下，首先先定义了一系列RPC所需的结构体：

```go
import (
	"fmt"
	"log"
	"net"
	"net/rpc"
	"sync"
)
// Common RPC request/reply definitions
const (
	OK       = "OK"
	ErrNoKey = "ErrNoKey"
)

type Err string

type PutArgs struct {
	Key   string
	Value string
}

type PutReply struct {
	Err Err
}

type GetArgs struct {
	Key string
}

type GetReply struct {
	Err   Err
	Value string
}
```

- 所有的服务器handler都定义了一个Args和Reply结构，客户端通过connect函数和服务器建立一个TCP连接，而get和put方法分别表示向服务器进行一次查询和存储，Call函数用于调用RPC服务，通过指定服务器函数的名字和相关的参数来实现RPC，具体的代码如下：

```go
func connect() *rpc.Client {
	client, err := rpc.Dial("tcp", ":1234")
	if err != nil {
		log.Fatal("dialing:", err)
	}
	return client
}

func get(key string) string {
	client := connect()
	args := GetArgs{"subject"}
	reply := GetReply{}
	err := client.Call("KV.Get", &args, &reply)
	if err != nil {
		log.Fatal("error:", err)
	}
	client.Close()
	return reply.Value
}

func put(key string, val string) {
	client := connect()
	args := PutArgs{"subject", "6.824"}
	reply := PutReply{}
	err := client.Call("KV.Put", &args, &reply)
	if err != nil {
		log.Fatal("error:", err)
	}
	client.Close()
}
```

- go语言需要服务器声明一个对象用作RPC处理，然后注册一系列的RPC服务，每当收到一个TCP链接的时候就将其交给RPC库来处理
- RPC库每次接受请求的时候会为其创建一个新的go-routine处理请求，并用生成的Reply对象作为返回值
- 服务器中的get和put方法用于在服务器中查询/添加一个key-value对，因此必须上锁，具体的代码如下：

```go
type KV struct {
	mu   sync.Mutex
	data map[string]string
}

func server() {
	kv := new(KV)
	kv.data = map[string]string{}
	rpcs := rpc.NewServer()
	rpcs.Register(kv)
	l, e := net.Listen("tcp", ":1234")
	if e != nil {
		log.Fatal("listen error:", e)
	}
	go func() {
		for {
			conn, err := l.Accept()
			if err == nil {
				go rpcs.ServeConn(conn)
			} else {
				break
			}
		}
		l.Close()
	}()
}

func (kv *KV) Get(args *GetArgs, reply *GetReply) error {
	kv.mu.Lock()
	defer kv.mu.Unlock()

	val, ok := kv.data[args.Key]
	if ok {
		reply.Err = OK
		reply.Value = val
	} else {
		reply.Err = ErrNoKey
		reply.Value = ""
	}
	return nil
}

func (kv *KV) Put(args *PutArgs, reply *PutReply) error {
	kv.mu.Lock()
	defer kv.mu.Unlock()

	kv.data[args.Key] = args.Value
	reply.Err = OK
	return nil
}
```

### RPC细节与故障处理

- 客户端和服务器的绑定：对于RPC来说，服务器的名字/端口号是一个必须的参数，一般来说大规模的系统会有专门用于配置和名字管理的服务器
- 处理请求中有一个Marshalling的过程，就是将一定格式的数据分装到一个packet里面，go语言的RPC库中可以传递数组，字符串，对象，映射等格式，一般来说将packet的指针传入函数中并进行Marshalling
- 一些可能存在的问题：丢包，网络故障，服务器崩溃，客户端长时间收不到响应。一个最简单的故障处理方式是“尽最大努力(best effort)”，每次call都稍微等待一段时间的回复并且如果没有收到回复就重新发送，重复几次相同的操作之后如果还没有收到response就结束
- 但是这种方法也存在很多问题，比如服务器一切正常只是发过来的包丢失了，就会造成重复操作的问题(比如将)，因此best effort只适用于只读的RPC操作
- 更好的故障处理方式应该对于每个RPC请求最多处理一次(at most once)，可以设立一定的机制来识别重复的请求，可以使用给请求标注ID等形式。
- 如果一个使用at most once机制的服务器也崩溃并且重启了，那么服务器就会忘记并且接受重复的恶请求，或许可以通过将重复请求的信息写到磁盘中以及使用多台服务器等方式来解决。
- go语言中的RPC就是一个简单形式的at most once，调用RPC的代码会在没有收到回复的情况下返回一个error