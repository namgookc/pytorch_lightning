import torch
import torch.utils.data as data
from torchvision.datasets import CIFAR10
from torchvision import transforms

import pytorch_lightning as pl

DATASET_PATH = '/home/ncho/Unzipped/0_prac/6_vtf/1_lightning/data'


def prep_loaders():
    # 1) transform
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    # 2) train-set & val-set
    train_dataset = CIFAR10(root=DATASET_PATH, train=True, transform=transform, download=True)
    pl.seed_everything(42)
    train_set, val_set = torch.utils.data.random_split(train_dataset, [45000,5000])

    # 3) test-set
    test_set = CIFAR10(root=DATASET_PATH, train=False, transform=transform, download=True)

    # 4) data-loaders
    train_loader = data.DataLoader(train_set, batch_size=256, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
    val_loader = data.DataLoader(val_set, batch_size=256, shuffle=False, drop_last=False, num_workers=4)
    test_loader = data.DataLoader(test_set, batch_size=256, shuffle=False, drop_last=False, num_workers=4)
    return train_loader, val_loader, test_loader


def get_train_images(num, train_dataset):
    return torch.stack([train_dataset[i][0] for i in range(num)], dim=0)
