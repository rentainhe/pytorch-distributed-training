import torch
import argparse
import torch.distributed as dist
from dataset import get_train_dataset, get_test_dataset
from model import resnet18

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
args = parser.parse_args()
print(args.local_rank)

dist.init_process_group(backend='nccl')

# t
#
#
# train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
#
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=..., sampler=train_sampler)