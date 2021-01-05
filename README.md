# pytorch-distributed-training
Distribute Dataparaller (DDP) Training on Pytorch

## Features
* Easy to study DDP training
* You can directly copy this code for a quick start

## Knowledge
### 1. Basic Theory
- __模型并行训练__ vs __数据并行训练__
  - 模型并行训练：整个网络的不同部分放在不同的GPU上进行训练，输入的数据相同，意思是每个GPU上只有模型的一部分。
  - 数据并行训练：不同的GPU输入不同的数据，运行相同的完整模型，意思是每个GPU上都有一个完整的模型同步在运行。
  - 模型并行通常会带来比较大的通信开销。

- __同步更新__ vs __异步更新__ （每个GPU负责一部分数据，涉及到参数更新问题时的两种不同处理方式）
  - 同步更新：每个batch所有GPU计算完成后，统一更新新的权值，然后所有GPU同步新的权值之后，再进行下一轮的计算
  - 异步更新：每个GPU计算完梯度之后，无需等待其他GPU更新，立即更新整体权值
  - 同步更新比异步更新更加稳定，且不会面临复杂的梯度问题，Loss抖动问题。在实践中更常用同步更新

- __Parameter Server__ vs __Ring All-Reduce__
  - Parameter Server：假设有4个GPU，`GPU 0`会将数据分成4份，分配到各个卡上，每一张卡负责自己那部分`mini—batch`的训练，得到grad后，返回给`GPU 0`做累积，得到更新的参数权重后，再传给各个卡上，更新模型权重
  - Ring All-Reduce：4张卡通过`环形`相连，每一张卡都与自己的`下一张卡`和`上一张卡`相连接，每个GPU会向自己的`下一张卡发送数据`，从自己的`上一张卡接受数据`
    - Ring All-Reduce 分为两个过程 `Scatter Reduce` 和 `All Gather`
    - Scatter Reduce：将参数量分为N份，假设参数总量为P，每一次传递的参数量为P/N，相邻的GPU传递不同的参数，传递N-1次之后可以得到每一份参数的累积
    - All Gather：累积得到参数之后，再进行一次传递，同步到每一个GPU上
  - 关于 Ring All-Reduce 以及其他分布式理论基础，在 [分布式训练（理论篇）](https://zhuanlan.zhihu.com/p/129912419) 里有很精彩的解读，这里不加赘述



## Implemented Work
参考的文章如下：
- [分布式训练（理论篇）](https://zhuanlan.zhihu.com/p/129912419)
- I learned from this [repo](https://github.com/tczhangzhi/pytorch-distributed), and want to make it easier and cleaner.
