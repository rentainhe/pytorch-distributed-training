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
from model import resnet18
from dataset import get_train_dataset, get_test_dataset

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def main():
    args = parser.parse_args()
    args.nprocs = torch.cuda.device_count()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    main_worker(args.local_rank, args.nprocs, args)

'''
    需要定义一个 main_worker，用来进行分布式训练，训练流程都需要写在 main_worker之中， main_worker就相当于每一个进程，通过传递的 local_rank 不同表示不同的进程
    main_worker 传递三个参数:
        - local_rank 当前进程的 index
        - nprocs 总共有多少个进程参与训练
        - args，自己指定的超参
'''
def main_worker(local_rank, nprocs, args):

    # 1. 分布式初始化，对于每一个进程都需要进行初始化，所以定义在 main_worker中
    cudnn.benchmark = True
    dist.init_process_group(backend='nccl')

    # 2. 基本定义，模型-损失函数-优化器
    model = resnet18()  # 定义模型，将对应进程放到对应的GPU上， .cuda(local_rank) / .set_device(local_rank)
    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])  # 将模型用 DistributedDataParallel 包裹
    criterion = nn.CrossEntropyLoss().cuda(local_rank)
    optimizer = torch.optim.SGD(model.parameters(), 0.1, momentum=0.9, weight_decay=1e-4)


    # 3. 加载数据，
    batch_size = int(args.batch_size / nprocs) # 需要手动划分 batch_size 为 mini-batch_size

    train_dataset = get_train_dataset()
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=2, pin_memory=True, sampler=train_sampler)

    test_dataset = get_test_dataset()
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=2, pin_memory=True, sampler=test_sampler)

    if args.evaluate:
        validate(test_loader, model, criterion, local_rank, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        test_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, local_rank, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, local_rank, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if args.local_rank == 0:
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.module.state_dict(),
                    'best_acc1': best_acc1,
                }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, local_rank, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), [batch_time, data_time, losses, top1, top5],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(local_rank, non_blocking=True)
        target = target.cuda(local_rank, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        torch.distributed.barrier()

        reduced_loss = reduce_mean(loss, args.nprocs)
        reduced_acc1 = reduce_mean(acc1, args.nprocs)
        reduced_acc5 = reduce_mean(acc5, args.nprocs)

        losses.update(reduced_loss.item(), images.size(0))
        top1.update(reduced_acc1.item(), images.size(0))
        top5.update(reduced_acc5.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, local_rank, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5], prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(local_rank, non_blocking=True)
            target = target.cuda(local_rank, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            torch.distributed.barrier()

            reduced_loss = reduce_mean(loss, args.nprocs)
            reduced_acc1 = reduce_mean(acc1, args.nprocs)
            reduced_acc5 = reduce_mean(acc5, args.nprocs)

            losses.update(reduced_loss.item(), images.size(0))
            top1.update(reduced_acc1.item(), images.size(0))
            top5.update(reduced_acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1**(epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()