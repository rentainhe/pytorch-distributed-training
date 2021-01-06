from torchvision import transforms, datasets

def get_train_dataset():

    transform_train = transforms.Compose([
        transforms.RandomCrop(32,padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
    ])
    trainset = datasets.CIFAR100(root='./data',
                                train=True,
                                download=True,
                                transform=transform_train)
    return trainset


def get_test_dataset():
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
    ])
    testset = datasets.CIFAR100(root='./data',
                                train=False,
                                download=True,
                                transform=transform_test)
    return testset
