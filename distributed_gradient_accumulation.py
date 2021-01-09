import argparse
import time
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
from utils.util import reduce_mean
from utils.validation import validate
import torch.optim as optim
import torch.multiprocessing as mp

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
parser.add_argument('--batch_size','--batch-size', default=256, type=int)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--ip', default='10.24.82.29', type=str)
parser.add_argument('--port', default='23456', type=str)
parser.add_argument('--grad_accu_steps', default=1, type=int, help="gradient accumulation steps")

def main():
    args = parser.parse_args()
    args.nprocs = torch.cuda.device_count()

    mp.spawn(main_worker, nprocs=args.nprocs, args=(args.nprocs, args))

'''
    需要定义一个 main_worker，用来进行分布式训练，训练流程都需要写在 main_worker之中， main_worker就相当于每一个进程，通过传递的 local_rank 不同表示不同的进程
    main_worker 传递三个参数:
        - local_rank 当前进程的 index
        - nprocs 总共有多少个进程参与训练
        - args，自己指定的超参
'''
def main_worker(local_rank, nprocs, args):
    args.local_rank = local_rank

    # 获得init_method的通信端口
    init_method = 'tcp://' + args.ip + ':' + args.port

    # 1. 分布式初始化，对于每一个进程都需要进行初始化，所以定义在 main_worker中
    cudnn.benchmark = True
    dist.init_process_group(backend='nccl', init_method=init_method, world_size=args.nprocs,
                            rank=local_rank)

    # 2. 基本定义，模型-损失函数-优化器
    model = resnet18()  # 定义模型，将对应进程放到对应的GPU上， .cuda(local_rank) / .set_device(local_rank)

    # 以下是需要加 local_rank 的部分：模型，损失函数
    # ================================
    torch.cuda.set_device(local_rank) # 使用 set_device 和 cuda 来指定需要的 GPU
    model.cuda(local_rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])  # 将模型用 DistributedDataParallel 包裹
    criterion = nn.CrossEntropyLoss().cuda(local_rank)
    # =================================
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=1e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

    # 3. 加载数据，
    batch_size = int(args.batch_size / nprocs) # 需要手动划分 batch_size 为 mini-batch_size

    train_dataset = get_train_dataset()
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, drop_last=True ,num_workers=4, pin_memory=True, sampler=train_sampler)

    test_dataset = get_test_dataset()
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, sampler=test_sampler)

    sub_batch_size = int(batch_size/args.grad_accu_steps)

    for epoch in range(args.epochs):
        start = time.time()
        model.train()
        # 需要设置sampler的epoch为当前epoch来保证dataloader的shuffle的有效性
        train_sampler.set_epoch(epoch)

        # 设置 train_scheduler 来调整学习率
        train_scheduler.step(epoch)
        loss_tmp = 0
        for step, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            for accu_step in range(args.grad_accu_steps):
                loss_tmp = 0
                # 将对应的sub_batch放到对应 GPU 上
                sub_images = images[accu_step * sub_batch_size:
                                    (accu_step + 1) * sub_batch_size]
                sub_images = sub_images.cuda(local_rank, non_blocking=True)
                sub_labels = labels[accu_step * sub_batch_size:
                                    (accu_step + 1) * sub_batch_size]
                sub_labels = sub_labels.cuda(local_rank, non_blocking=True)

                outputs = model(sub_images)
                loss = criterion(outputs, sub_labels)
                loss_tmp += loss
                loss = loss / args.grad_accu_steps
                loss.backward()

            # torch.distributed.barrier()的作用是，阻塞进程，保证每个进程运行完这一行代码之前的所有代码，才能继续执行，这样才计算平均loss和平均acc的时候不会出现因为进程执行速度不一致的错误
            torch.distributed.barrier()
            reduced_loss = reduce_mean(loss_tmp, args.nprocs)

            # 更新优化模型权重
            optimizer.step()

            if args.local_rank == 0:
                print(
                    'Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                        reduced_loss,
                        optimizer.param_groups[0]['lr'],
                        epoch=epoch+1,
                        trained_samples=step * args.batch_size + len(images),
                        total_samples=len(train_loader.dataset)
                    ))

        finish = time.time()
        if args.local_rank == 0:
            print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

        # validate after every epoch
        validate(test_loader, model, criterion, local_rank, args)


if __name__ == '__main__':
    main()