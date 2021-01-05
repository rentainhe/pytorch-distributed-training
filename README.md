# pytorch-distributed-training
Distribute Dataparallel (DDP) Training on Pytorch

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

### 2. Basic Concept
- group: 表示进程组，默认情况下只有一个进程组。
- world size: 全局进程个数
- rank: 进程序号，用于进程间通讯，表示进程优先级，`rank=0`表示`主进程`
- local_rank: 进程内，`GPU`编号，非显示参数，由`torch.distributed.launch`内部指定，`rank=3, local_rank=0` 表示第`3`个进程的第`1`块`GPU`

### 3. API Introduction
- 库文件导入
```python
import datetime
import torch.distributed as dist
```

- 初始化进程组：`init_process_group`
```python
import datetime
import torch.distributed as dist
dist.init_process_group(backend='nccl',
                        init_method=None,
                        timeout=datetime.timedelta(0, 1800),
                        world_size=-1,
                        rank=-1,
                        store=None)
```
__function__: 每个进程中调用该函数，用于初始化该进程。在使用DDP时，该函数需要在`distributed`内所有相关函数之前使用

- `backend`: 指定当前进程所需使用的`通信后端`，`小写字符串`，支持的通信后端有 `gloo`, `mpi`, `nccl`。__CPU__ 上使用`gloo`和`mpi`, __GPU__ 上建议使用`nccl` 
- `init_method`: 指定当前进程组的初始化方式，可选参数，与`store`参数互斥
  - TCP初始化: `init_method='tcp://ip:port'`, ip表示主节点的ip地址, rank=0的主机的ip地址, 然后再选择一个闲置的端口号即可
  - 共享文件系统: `init_method='file:///mnt/nfs/sharedfile'`, 这个初始化方法比较麻烦，提供的共享文件`在一开始的时候不应存在`，这个方法在结束时也`不会自动删除共享文件`，所以在每次使用时应该`手动删除`上次的`自动创建`的共享文件
- `rank`: rank表示当前进程的优先级。如果指定了`store`参数，则必须指定该参数，`rank=0`表示主进程，即`master`节点。
- `world_size`: 进程总数，如果指定了`store`参数，则需要指定该参数
- `timeout`: 指定每个进程的超时时间，可选参数，`datetime.timedelta`对象，默认为`30`分钟，仅用于`gloo`后端
- 所有`worker`可访问的`key / value`，用于交换连接/地址信息。与`init_method`互斥。

### Usage
#### 1. 获取当前进程的index
pytorch可以通过torch.distributed.lauch启动器，在命令行分布式地执行.py文件, 在执行的过程中会将当前进程的index通过参数传递给python
```python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
args = parser.parse_args()
print(args.local_rank)
```
#### 2. 使用init_process_group 
设置GPU之间通信使用的后端和端口
```python
import torch.distributed as dist
dist.init_process_group(backend='nccl')
```
#### 3. 使用DistributedSampler对数据集进行划分
可以将每个batch分配成几个mini-batch分配给不同进程进行训练
```python
import torch
train_dataset = ...
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=..., num_workers=..., pin_memory=..., 
                                          sampler=train_sampler)

test_dataset = ...
test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
testloader = torch.utils.data.DataLoader(train_dataset, batch_size=..., num_workers=..., pin_memory=..., 
                                         sampler=test_sampler) 
```
#### 4. 使用 DistributedDataParallel包装模型
可以帮助我们将不同GPU上求得的梯度进行all-reduce操作，不同GPU模型的梯度均为all-reduce之前GPU梯度的均值
```python
import torch
from torchvision.models import resnet18
model = resnet18()
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
```

#### 5. 将模型加载到当前进程使用的GPU中，正常进行反向传播
```python
torch.cuda.set_device(args.local_rank)

model.cuda()

for epoch in range(100):
   for batch_idx, (data, target) in enumerate(train_loader):
      images = images.cuda(non_blocking=True)
      target = target.cuda(non_blocking=True)
      ...
      output = model(images)
      loss = criterion(output, target)
      ...
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
```

#### 6. Command Line 启动
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main.py
```

## Details
### 1. DistributedSampler
在 `Pytorch1.2` 版本之前，DistributedSampler没有`shuffle`参数，在 `Pytorch1.2` 之后出现`shuffle`参数，但是在采样数据的时候，shuffle使用的随机种子等于当前的epoch

如果不手动修改 epoch 的话，每一轮迭代，shuffle的顺序相同，起不到shuffle的作用，如果想使用shuffle的功能，需要手动设置`DistributedSampler`的epoch
```python
import torch
batch_size = 128

train_dataset = ...
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=2, pin_memory=True, sampler=train_sampler)

test_dataset = ...
test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=2, pin_memory=True, sampler=test_sampler)

for epoch in range(args.epochs):
    train_sampler.set_epoch(epoch)
    test_sampler.set_epoch(epoch)
    ...
```

## Implemented Work
参考的文章如下：
- [分布式训练（理论篇）](https://zhuanlan.zhihu.com/p/129912419)
- [DistributedSampler的问题](https://www.zhihu.com/question/67209417/answer/1017851899)
- I learned from this [repo](https://github.com/tczhangzhi/pytorch-distributed), and want to make it easier and cleaner.
