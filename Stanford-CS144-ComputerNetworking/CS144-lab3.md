# CS144-ComputerNetworking-Lab3

> Stanford CS144 Introduction to Computer Network课程Project的lab3，这个lab主要任务是实现一个TCP的接收端TCPSender



## Overview

书接上回，上一个lab中我们实现了TCP接收端的基本功能，而这一次我们需要实现的是TCP的发送端，发送端的核心类TCPsender的主要作用是将发送端存储的字节流分割成若干个TCPSegment并将其发送出去，同时在发送的过程中还要做到流量控制和丢失重传等功能，它的功能可以概括成以下几个点：

- 关注Receiver端的窗口大小，即处理接收端发送过来的ackno和window size
- 在发送数据的时候，根据窗口的大小尽可能的从存储数据流的ByteStream中读取对应数量的字节并生成对应的TCP Segment然后将它们发送出去
- 关注哪些segment发送出去之后没有收到回复，这些segment被称为outstanding的，并且如果一个segment发送出去之后超过一定的时间还没有收到回复，那么它就会被重新发送

> 这个操作被称为Automatic Repeat Request，发送方会根据接收方的窗口大小尽可能多的发送segment，并且要保证接收端接收到字节流中的每个字节至少一次，所以对于超过一定时间的segment，如果还不能确认它已经被接收，那么就会重新发送。

## Sender怎么判断segment丢失

TCPSender发送的segment的数据部分来自ByteStream中存储的可信字节流，并且每个字节都有专属的index，同时TCP的头部还会有SYN和FIN标记来表明这个segment的一些信息。

同时，Sender必须在发送之后继续追踪每一个发出的segment直到收到来自Receiver的回复，因为收到了Receiver回复的ackno和window size才能判断某段数据是否已经被接收。Sender会阶段性的调用tick函数来了解距离上次调用tick过去的时间，并且Sender中会维护一个记录已经发送出去的segment的结构，并且每次将其中最老的且没有收到回复的数据包重新发送一次。

- 这里的没有被接收指的是这个segment没有任何一个字节被接收端所接收。

TCPSender这个类对于超时重传的具体规定和实现要求如下：

- 每过若干毫秒，tick函数都会被调用，并且会传入一个参数，表示距离上次调用tick已经有多少秒过去。通过这种方式，TCPSender可以维护一个自己已经存在的时长的信息，并且我们不能在sponge的代码中调用clock和time相关的函数，tick是TCPSender获知时间信息的唯一途径
- 当一个TCPSender对象被构造的时候，会有一个重传时间(Retransmission timeout, RTO)的初始值作为构造函数的参数，RTO代表了对于一个outstanding segment的等待时间，也就是超过RTO还没有收到ackno的segment会被认为是丢失了
- 我们需要实现一个重传计时器timer，这个计时器可以在任何时候开启，并且记录经过的时间，并且计时器通过tick函数来获知时间的流逝
- 每一次有非空segment被发送，如果计时器没有启动就需要开启计时器，这样一来计时器就会记录segment发送出去所经过的时间，当所有outstanding segment都发送成功之后，计时器才会停止
- 当tick函数被调用，并且计时器发现超时的时候，就会进行如下操作：
  - 将最早发送出去并且还没有收到ackno的segment的segment进行重传
  - 如果接收端的窗口大小是非零的，那么就要：
    - 记录**连续重传**(consecutive retransmission)的次数，后面的lab中将会用到这个重要的信息，将被用来判断当前的TCP连接还是否有效
    - 将RTO的值翻倍，这个操作被称为**指数退避**(exponential backoff)，这么做可以降低重传的频率，减小网络的负载量
  - **重新设置**重传计时器记录的时间，等待下一次超过RTO
- 当Receiver给Sender一个ackno的时候，Sender需要
  - 将RTO设置回原本的初始值
  - 如果Sender还有outstanding segment就重开重传计时器
  - 将连续重传数重置为0

我们可以发现，这样一套运行逻辑中，RTO的值会根据ackno收到的情况不断变化，如果长时间收不到ackno，那么Sender会不断进行重传，并且增大RTO的时间，而如果收到了新的ackno就会把扩大的RTO重置回初始值。

## 代码实现

### 类的定义

首先我们需要定义一些类实现过程中要用到的类变量

```c++
class TCPSender {
  private:
    //! our initial sequence number, the number for our SYN.
    WrappingInt32 _isn;

    //! outbound queue of segments that the TCPSender wants sent
    std::queue<TCPSegment> _segments_out{};

    //! retransmission timer for the connection
    unsigned int _initial_retransmission_timeout;

    //! outgoing stream of bytes that have not yet been sent
    ByteStream _stream;

    //! the (absolute) sequence number for the next byte to be sent
    uint64_t _next_seqno{0};

    std::map<uint64_t, TCPSegment> _outstanding_segs{};
    std::optional<uint64_t> _timer_ms = std::nullopt;
    uint64_t _ms = 0, _rto, _bytes_in_flight = 0;
    uint32_t _consecutive_retransmissions = 0;
    std::optional<uint64_t> _last_ackno = std::nullopt;
    std::optional<uint16_t> _last_window_size = 1;
    bool _fin_sent = false, _zero_window_size = false;
}
```

这里发送出去的segment用一个队列来存储，要发送的时候就push到队列里面去，而对outstanding segment的情况追踪则使用一个map来存储，方便进行查询，另外还要设置像是timer之类的上面提到过的变量。

### send_empty_segment

这个函数主要用来发送一个空的segment，虽然这个lab里面用不上这个东西，但是据说后面的lab4要用，所以lab3也要求实现一下，其实就是新建一个TCPSegment对象，然后改变一下对应的属性值比如seqno就行：

```c++
void TCPSender::send_empty_segment() {
    TCPSegment seg;
    seg.header().seqno = wrap(_next_seqno, _isn);
    seg.header().syn = seg.header().fin = false;
    seg.payload() = Buffer("");
    _segments_out.push(seg);
}
```



### tick

tick的作用是进行时间的计数，并且观察时间是否相比于timer超过了RTO，如果超过了就需要进行重传的操作，重传就是要将第一个outstanding segment取出(这里的outstanding segment是个map，并且key是seqno，所以map的第一个就是seqno最小的那个，也就是最早发出去的那个)，然后还要记录连续重传的次数。

```c++
void TCPSender::tick(const size_t ms_since_last_tick) {
    _ms += ms_since_last_tick;
    if (!_timer_ms.has_value()) {
        return;
    }
    // retransmission
    if (_ms - *_timer_ms >= _rto) {
        _timer_ms = _ms;
        auto seg = _outstanding_segs.begin()->second;
        _segments_out.push(seg);
        if (!_zero_window_size) {
            _consecutive_retransmissions += 1;
            _rto *= 2;
        }
    }
}
```



### fill_window

fill_window这个函数的主要作用是创建一个大小跟当前的窗口大小刚好一样大的segment并发送出去，我们首先要判断这个segment的SYN和FIN两个标记的情况，然后从ByteStream取出数量合适的字节填充进payload里面，再将这个新的segment加入发送队列中，同时在outstanding segment中也记录它的信息。

```c++
void TCPSender::fill_window() {
    while (true) {
        bool syn =(_next_seqno == 0);
        bool fin = _stream.input_ended() && _stream.buffer_size() <= TCPConfig::MAX_PAYLOAD_SIZE &&
                   (static_cast<int64_t>(_stream.buffer_size()) <= static_cast<int64_t>(*_last_window_size) - syn - 1);
        // size of the payload to fill the window.
        uint64_t payload_size = std::min({TCPConfig::MAX_PAYLOAD_SIZE,
                                      _stream.buffer_size(),
                                      static_cast<uint64_t>(*_last_window_size - syn - fin)});
        TCPSegment seg;
        seg.header().seqno = wrap(_next_seqno, _isn);
        seg.header().syn = syn;
        seg.header().fin = fin;
        seg.payload() = Buffer( _stream.read(payload_size));
        if (_fin_sent || (syn == false && fin == false && payload_size == 0)) {
            return;
        }
        if (fin) {
            _fin_sent = true;
        }
        // send out the TCP segment.
        _segments_out.push(seg);
        _outstanding_segs.emplace(_next_seqno, seg);
        size_t seg_length = seg.length_in_sequence_space();
        _bytes_in_flight += seg_length;
        _next_seqno += seg_length;
        *_last_window_size -= seg_length;
        if (!_timer_ms.has_value()) {
            _timer_ms = _ms;
        }
    }
    
}
```



### ack_received

ack_received这个函数的主要功能是接收来自Receiver的回复，这个回复包括了ackno和window size两个信息，然后我们需要将outstanding segment中被ackno确认的segment都移除，并记录正确的window size，然后还需要将RTO和重传次数设置成初始值。

```c++
void TCPSender::ack_received(const WrappingInt32 ackno, const uint16_t window_size) {
    uint64_t ack = unwrap(ackno, _isn, _last_ackno.has_value() ? *_last_ackno : 0);
    bool new_data = false;
    if (_last_ackno.has_value()) {
        if (_last_ackno <= ack && ack <= _next_seqno) {
            new_data = ack > _last_ackno;
            _last_ackno = ack;
            this->_zero_window_size = window_size == 0;
            uint16_t new_window_size = this->_zero_window_size ? 1 : window_size;
            _last_window_size = static_cast<uint64_t>(
            std::max(0l, static_cast<int64_t>(ack) + new_window_size - static_cast<int64_t>(_next_seqno)));
        }
    } else if (ack <= _next_seqno) {
        new_data = true;
        _last_ackno = ack;
        this->_zero_window_size = window_size == 0;
        uint16_t new_window_size = this->_zero_window_size ? 1 : window_size;
        _last_window_size = static_cast<uint64_t>(
            std::max(0l, static_cast<int64_t>(ack) + new_window_size - static_cast<int64_t>(_next_seqno)));
    }
    if (!new_data) {
        return;
    }
    std::vector<std::pair<uint64_t, TCPSegment>> segments;
    for (auto kv : _outstanding_segs) {
        TCPSegment segment = kv.second;
        uint64_t seqno = kv.first;
        if (seqno + segment.length_in_sequence_space() > _last_ackno) {
            segments.emplace_back(seqno, segment);
        }
    }
    _outstanding_segs.clear();
    _bytes_in_flight = 0;
    for (const auto &kv : segments) {
        _outstanding_segs.insert(kv);
        _bytes_in_flight += kv.second.length_in_sequence_space();
    }

    _rto = _initial_retransmission_timeout;
    _consecutive_retransmissions = 0;
    if (_outstanding_segs.empty()) {
        _timer_ms = std::nullopt;
    } else {
        _timer_ms = _ms;
    }
}
```

