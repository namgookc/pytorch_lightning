import torch
import torch.utils.data as data
from torchvision.datasets import CIFAR10
from torchvision import transforms

from utils import set_seed

DATASET_PATH = '/home/ncho/Unzipped/0_prac/6_vtf/1_lightning/data'


def calculate_whitening():
    train_dataset = CIFAR10(root=DATASET_PATH, train=True, download=True)
    DATA_MEANS = (train_dataset.data/255.0).mean(axis=(0,1,2))
    DATA_STD = (train_dataset.data/255.0).std(axis=(0,1,2))
    return DATA_MEANS, DATA_STD


def prep_loaders(data_mean, data_std):
    # 1) transforms
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.RandomResizedCrop((32,32), scale=(0.8,1.0), ratio=(0.9,1.1)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(data_mean,data_std)])
    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(data_mean,data_std)])

    # 2) train-set & val-set
    train_dataset = CIFAR10(root=DATASET_PATH, train=True, transform=train_transform, download=True)
    val_dataset = CIFAR10(root=DATASET_PATH, train=True, transform=test_transform, download=True)
    set_seed(42)
    train_set, _ = torch.utils.data.random_split(train_dataset, [45000,5000])
    set_seed(42)
    _, val_set = torch.utils.data.random_split(val_dataset, [45000,5000])
    train_loader = data.DataLoader(train_set, batch_size=128, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
    val_loader = data.DataLoader(val_set, batch_size=128, shuffle=False, drop_last=False, num_workers=4)

    # 3) test-set
    test_set = CIFAR10(root=DATASET_PATH, train=False, transform=test_transform, download=True)
    test_loader = data.DataLoader(test_set, batch_size=128, shuffle=False, drop_last=False, num_workers=4)

    # 4) verification
    if 0:
        imgs, _ = next(iter(train_loader))
        print('Batch mean', imgs.mean(dim=[0,2,3]))
        print('Batch std', imgs.std(dim=[0,2,3]))
    return train_loader, val_loader, test_loader
