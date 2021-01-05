import csv

import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from model import resnet18
from dataset import get_train_dataset, get_test_dataset

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='/home/zhangzhi/Data/exports/ImageNet2012', help='path to dataset')
parser.add_argument('--epochs', default=90, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')

def main():
    args = parser.parse_args()
    args.nprocs = torch.cuda.device_count()

    mp.spawn(main_worker, nprocs=args.nprocs, args=(args.nprocs, args))

def main_worker(local_rank, nprocs, args):
    '''
    :param local_rank: 对应当前进程的index
    :param nprocs: 对应进程数量
    :param args: 对应其他超参数
    '''
    args.local_rank = local_rank

    best_acc = .0

    # 1. Pytorch Distributed 初始化 参数解析
    # backend：后端选择，gpu一般选用 'nccl'，cpu一般选用 'gloo'， 强烈建议使用 nccl
    # init_method：初始化包的URL，用作并发控制的共享方式
    # world_size：参与工作的进程数
    # rank：当前进程的rank
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:123456', world_size=args.nprocs, rank=local_rank)

    # create model
    model = resnet18()
    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)

    # divide the batch_size
    args.batch_size = int(args.batch_size / args.nprocs)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # define loss function
    criterion = nn.CrossEntropyLoss().cuda(local_rank)
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    cudnn.benchmark = True

    train_dataset = get_train_dataset()
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=2,
                                               pin_memory=True,
                                               sampler=train_sampler)

    val_dataset = get_test_dataset()
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=2,
                                             pin_memory=True,
                                             sampler=val_sampler)
