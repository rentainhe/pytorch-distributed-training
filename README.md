## pytorch-distributed-training
Distribute Dataparallel (DDP) Training on Pytorch

### Features
* Easy to study DDP training
* You can directly copy this code for a quick start
* Learning Notes Sharing(with `√`means finished):
  - [x] [Basic Theory](https://github.com/rentainhe/pytorch-distributed-training/blob/master/tutorials/0.%20Basic%20Theory.md)
  - [x] [Pytorch Gradient Accumulation](https://github.com/rentainhe/pytorch-distributed-training/blob/master/tutorials/1.%20Gradient%20Accumulation.md)
  - [x] [More Details of DDP Training](https://github.com/rentainhe/pytorch-distributed-training/blob/master/tutorials/2.%20DDP%20Training%20Details.md)
  - [x] [DDP training with apex](https://github.com/rentainhe/pytorch-distributed-training/blob/master/tutorials/4.%20DDP%20with%20apex.md)
  - [ ] [Accelerate-on-Accelerate DDP Training Tricks](https://github.com/rentainhe/pytorch-distributed-training/blob/master/tutorials/3.%20DDP%20Training%20Tricks.md)
  - [ ] [DP and DDP 源码解读](https://github.com/rentainhe/pytorch-distributed-training/blob/master/tutorials/5.%20DP%20and%20DDP.md)

### Good Notes
分享一些网上优质的笔记
- [分布式训练（理论篇）](https://zhuanlan.zhihu.com/p/129912419)
- [当代研究生应当掌握的并行训练方法（单机多卡）](https://zhuanlan.zhihu.com/p/98535650)

### TODO
- [ ] 完成DP和DDP源码解读笔记(当前进度50%)
- [ ] 修改代码细节, 复现实验结果

### Quick start
想直接运行查看结果的可以执行以下命令, 注意一定要用`--ip`和`--port`来指定主机的`ip`地址以及空闲的`端口`，否则可能无法运行
- [dataparaller.py](https://github.com/rentainhe/pytorch-distributed-training/blob/master/dataparallel.py)
```bash
$ python dataparallel.py --gpu 0,1,2,3
```

- [distributed.py](https://github.com/rentainhe/pytorch-distributed-training/blob/master/distributed.py)
```bash
$ CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 distributed.py
```

- [distributed_mp.py](https://github.com/rentainhe/pytorch-distributed-training/blob/master/distributed_mp.py)
```bash
$ CUDA_VISIBLE_DEVICES=0,1,2,3 python distributed_mp.py
```

- [distributed_apex.py](https://github.com/rentainhe/pytorch-distributed-training/blob/master/distributed_apex.py)
```bash
$ CUDA_VISIBLE_DEVICES=0,1,2,3 python distributed_apex.py
```

- `--ip=str`, e.g `--ip='10.24.82.10'` 来指定主进程的ip地址
- `--port=int`, e.g `--port=23456` 来指定启动端口号
- `--batch_size=int`, e.g `--batch_size=128` 设定训练batch_size

- [distributed_gradient_accumulation.py](https://github.com/rentainhe/pytorch-distributed-training/blob/master/distributed_gradient_accumulation.py)
```bash
$ CUDA_VISIBLE_DEVICES=0,1,2,3 python distributed_apex.py
```
- `--ip=str`, e.g `--ip='10.24.82.10'` 来指定主进程的ip地址
- `--port=int`, e.g `--port=23456` 来指定启动端口号
- `--grad_accu_steps=int`, e.g `--grad_accu_steps=4'` 来指定gradient_step


### Comparison
结果不够准确，GPU状态不同结果可能差异较大

默认情况下都使用`SyncBatchNorm`, 这会导致执行速度变慢一些，因为需要增加进程之间的通讯来计算`BatchNorm`, 但有利于保证准确率

Concepts
- [apex](https://github.com/NVIDIA/apex)
- DP: `DataParallel`
- DDP: `DistributedDataParallel`

Environments
- 4 × 2080Ti

|model|dataset|training method|time(seconds/epoch)|Top-1 accuracy
|:---:|:---:|:---:|:---:|:---:
|resnet18|cifar100|DP|20s|
|resnet18|cifar100|DP+apex|18s|
|resnet18|cifar100|DDP|16s|
|resnet18|cifar100|DDP+apex|14.5s|

### Basic Concept
- group: 表示进程组，默认情况下只有一个进程组。
- world size: 全局进程个数
  - 比如16张卡`单卡单进程`: world size = 16
  - `8卡单进程`: world size = 1
  - 只有当连接的进程数等于world size, 程序才会执行
- rank: 进程序号，用于进程间通讯，表示进程优先级，`rank=0`表示`主进程`
- local_rank: 进程内，`GPU`编号，非显示参数，由`torch.distributed.launch`内部指定，`rank=3, local_rank=0` 表示第`3`个进程的第`1`块`GPU`


### Usage 单机多卡
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
#### 2. 定义 main_worker 函数 
主要的训练流程都写在main_worker函数中，main_worker需要接受三个参数（最后一个参数optional）: 
```python
def main_worker(local_rank, nprocs, args):
    training...
```
- local_rank: 接受当前进程的rank值，在一机多卡的情况下对应使用的GPU号
- nprocs: 进程数量
- args: 自己定义的额外参数

main_worker,相当于你每个进程需要运行的函数（每个进程执行的函数内容是一致的，只不过传入的local_rank不一样）

#### 3. main_worker函数中的整体流程
main_worker函数中完整的训练流程
```python
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
def main_worker(local_rank, nprocs, args):
    args.local_rank = local_rank
    # 分布式初始化，对于每个进程来说，都需要进行初始化
    cudnn.benchmark = True
    dist.init_process_group(backend='nccl', init_method='tcp://ip:port', world_size=nprocs, rank=local_rank)
    # 模型、损失函数、优化器定义
    model = ...
    criterion = ...
    optimizer = ...
    # 设置进程对应使用的GPU
    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)
    # 使用分布式函数定义模型
    model = model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    
    # 数据集的定义，使用 DistributedSampler
    mini_batch_size = batch_size / nprocs # 手动划分 batch_size to mini-batch_size
    train_dataset = ...
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=mini_batch_size, num_workers=..., pin_memory=..., 
                                              sampler=train_sampler)
    
    test_dataset = ...
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    testloader = torch.utils.data.DataLoader(train_dataset, batch_size=mini_batch_size, num_workers=..., pin_memory=..., 
                                             sampler=test_sampler) 
    
    # 正常的 train 流程
    for epoch in range(300):
       model.train()
       for batch_idx, (images, target) in enumerate(trainloader):
          images = images.cuda(non_blocking=True)
          target = target.cuda(non_blocking=True)
          ...
          pred = model(images)
          loss = loss_function(pred, target)
          ...
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
```

#### 4. 定义main函数
```python
import argparse
import torch
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--batch_size','--batch-size', default=256, type=int)
parser.add_argument('--lr', default=0.1, type=float)

def main_worker(local_rank, nprocs, args):
    ...

def main():
    args = parser.parse_args()
    args.nprocs = torch.cuda.device_count()
    # 执行 main_worker
    main_worker(args.local_rank, args.nprocs, args)

if __name__ == '__main__':
    main()
```

#### 5. Command Line 启动
```bash
$ CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 distributed.py
```

- `--ip=str`, e.g `--ip='10.24.82.10'` 来指定主进程的ip地址
- `--port=int`, e.g `--port=23456` 来指定启动端口号

参数说明:
- --nnodes 表示机器的数量
- --node_rank 表示当前的机器
- --nproc_per_node 表示每台机器上的进程数量

参考 [distributed.py](https://github.com/rentainhe/pytorch-distributed-training/blob/master/distributed.py)

#### 6. torch.multiprocessing 
使用`torch.multiprocessing`来解决进程自发控制可能产生问题，这种方式比较稳定，推荐使用
```python
import argparse
import torch
import torch.multiprocessing as mp

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--batch_size','--batch-size', default=256, type=int)
parser.add_argument('--lr', default=0.1, type=float)

def main_worker(local_rank, nprocs, args):
    ...

def main():
    args = parser.parse_args()
    args.nprocs = torch.cuda.device_count()
    # 将 main_worker 放入 mp.spawn 中
    mp.spawn(main_worker, nprocs=args.nprocs, args=(args.nprocs, args))

if __name__ == '__main__':
    main()
```


参考 [distributed_mp.py](https://github.com/rentainhe/pytorch-distributed-training/blob/master/distributed_mp.py) 启动方式如下:
```bash
$ CUDA_VISIBLE_DEVICES=0,1,2,3 python distributed_mp.py
```

- `--ip=str`, e.g `--ip='10.24.82.10'` 来指定主进程的ip地址
- `--port=int`, e.g `--port=23456` 来指定启动端口号


## Reference
参考的文章如下（如果有文章没有引用，但是内容差不多的，可以提issue给我，我会补上，实在抱歉）：
- [Pytorch: DDP系列](https://zhuanlan.zhihu.com/p/178402798)
- [分布式训练](https://zhuanlan.zhihu.com/p/98535650)
- [分布式训练（理论篇）](https://zhuanlan.zhihu.com/p/129912419)
- [DistributedSampler的问题](https://www.zhihu.com/question/67209417/answer/1017851899)
- I learned from this [repo](https://github.com/tczhangzhi/pytorch-distributed), and want to make it easier and cleaner.
