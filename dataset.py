from torchvision import transforms, datasets

def get_train_dataset():

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = datasets.CIFAR100(root='./data',
                                train=True,
                                download=True,
                                transform=transform_train)
    return trainset


def get_test_dataset():
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    testset = datasets.CIFAR100(root='./data',
                                train=False,
                                download=True,
                                transform=transform_test)
    return testset
