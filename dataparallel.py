import argparse
import time
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from utils.model import resnet18
from utils.dataset import get_train_dataset, get_test_dataset
from utils.util import AverageMeter, ProgressMeter, accuracy
import torch.optim as optim


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
parser.add_argument('--batch_size','--batch-size', default=256, type=int)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--gpu', type=str, default='0' ,help="gpu choose, eg. '0,1,2,...' ")

def main():
    args = parser.parse_args()
    args.nprocs = torch.cuda.device_count()
    # set training gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    n_gpu = len(args.gpu.split(','))
    gpus = [ _ for _ in range(n_gpu) ]

    main_worker(gpus=gpus, args=args)


def main_worker(gpus, args):
    torch.cuda.set_device('cuda:{}'.format(gpus[0]))

    # 定义模型，损失函数，优化器
    model = resnet18()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=1e-4)

    model.cuda()
    # 如果使用的GPU数量大于1，需要用nn.DataParallel来修饰模型
    if len(gpus) > 1:
        model = nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])

    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)


    train_dataset = get_train_dataset()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    test_dataset = get_test_dataset()
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    for epoch in range(args.epochs):
        start = time.time()
        model.train()

        # 设置 train_scheduler 来调整学习率
        train_scheduler.step(epoch)

        for step, (images, labels) in enumerate(train_loader):
            # 将对应进程的数据放到 GPU 上
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)


            # 更新优化模型权重
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(
                'Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                    loss,
                    optimizer.param_groups[0]['lr'],
                    epoch=epoch+1,
                    trained_samples=step * args.batch_size + len(images),
                    total_samples=len(train_loader.dataset)
                ))

        finish = time.time()
        print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

        # validate after every epoch
        validate(test_loader, model, criterion)

def validate(val_loader, model, criterion):
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
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg


if __name__ == '__main__':
    main()