import argparse
import torchvision.models as models
from model import resnet18
from dataset import get_train_dataset, get_test_dataset

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))
print(model_names)

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='/home/zhangzhi/Data/exports/ImageNet2012', help='path to dataset')
parser.add_argument('-a',
                    '--arch',
                    metavar='ARCH',
                    default='resnet18',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')