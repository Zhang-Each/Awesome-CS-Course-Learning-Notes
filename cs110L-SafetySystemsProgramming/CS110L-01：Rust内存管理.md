# CS110L-01：Rust内存管理

## 从一个C语言的例子开始

下面是一段C语言编写的代码，存在着一大堆内存安全问题，请欣赏：

```C
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

typedef struct {
  int* data;     // Pointer to our array on the heap
  int  length;   // How many elements are in our array
  int  capacity; // How many elements our array can hold
} Vec;

Vec* vec_new() {
  Vec vec;
  vec.data = NULL;
  vec.length = 0;
  vec.capacity = 0;
  return &vec;
}

void vec_push(Vec* vec, int n) {
  if (vec->length == vec->capacity) {
    int new_capacity = vec->capacity * 2;
    int* new_data = (int*) malloc(new_capacity);
    assert(new_data != NULL);

    for (int i = 0; i < vec->length; ++i) {
      new_data[i] = vec->data[i];
    }

    vec->data = new_data;
    vec->capacity = new_capacity;
  }

  vec->data[vec->length] = n;
  ++vec->length;
}

void vec_free(Vec* vec) {
  free(vec);
  free(vec->data);
}

void main() {
  Vec* vec = vec_new();
  vec_push(vec, 107);

  int* n = &vec->data[0];
  vec_push(vec, 110);
  printf("%d\n", *n);

  free(vec->data);
  vec_free(vec);
}
```

据说这一段代码里有7个内存相关的安全问题，比如：

- `vec_new`函数中最后返回的vec在函数结束的时候已经被销毁，不能作为结果返回，因此出现了Out of Memory的问题
- main函数中出现了重复的内存释放
- `vec_push`中没有释放`vec_data`，会出现内存泄漏的问题

这些内存管理上的问题可以通过在编译期加强对程序代码的检查来尽量避免，而Rust正是通过这种方式尽可能减少了此类内存管理问题的发生。同时Rust有的时候性能甚至会超过同样逻辑的C语言代码，因为编译器在编译阶段对程序作出了非常多的优化。

## Rust的内存管理机制

Rust针对内存管理可能出现的各种问题提出了所有者机制(Ownership)和生命周期的概念

为了避免变量的使用出现各种意想不到的chaos，我们应该尽量避免对变量的同时操作，一个变量应该只允许被一个引用进行修改或者多个引用同时读取。这就是Rust的Borrowing机制，一个变量值可以**同时有多个不可变的共享引用**，但是可变引用有且只能有一个

同时每个值有其生命周期，当一个值被创建的时候其生命周期就开始了，而当这段代码块执行结束的时候这个值也就消失了，Rust**不允许对已经消失的值继续进行引用**，一旦出现就会产生编译上的错误。所以要特别注意，一个对象被函数调用之后，如果函数的参数格式不是引用，那么函数结束之后这个对象对应的值的生命周期就到头了，后续就不能再被引用，所以写Rust的函数的时候一定要想清楚参数类型要不要加引用，加上引用这个对象在函数执行之后就不会消灭，可以继续使用，比如看下面这个例子

```rust
fn change_it_up(s: &mut String) {
 *s = "goodbye".to_string();
}
fn make_it_plural(word: &mut String) {
 word.push('s');
}
fn let_me_see(s: &String) {
 println!("{}", s);
}
fn main() {
 let mut s = "hello".to_string();
 change_it_up(&mut s);
 let_me_see(&s);
 make_it_plural(&mut s);
 let_me_see(&s);
 // let's make it even more plural
 s.push('s'); // does this seem strange?
 let_me_see(&s);
}
```

这个例子中定义的函数都在参数类型声明中用了`&mut` 

同时，Rust在定义变量的时候需要用**关键字mut**来指定变量是可改变的变量，默认的变量都是不可变的，这和C/C++中的const正好相反。

这些编译期的检查和安全保证应该说是Rust最核心的特性，当然我也似懂非懂，很多概念解释不清楚，只能说看起来好像是这样。

