# -*- coding: utf-8 -*-
import os
import sys
from contextlib import contextmanager

import torch
import torchvision
import torchvision.transforms as transforms


@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


transform_train = transforms.Compose(
    [transforms.RandomHorizontalFlip(p = 0.5),
     transforms.RandomCrop(32, padding = 4),
     transforms.RandomGrayscale(),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

transform_test = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

with suppress_stdout():
    train_set = torchvision.datasets.CIFAR10(root = './CIFAR10', train = True,
                                             download = True, transform = transform_train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = 100,
                                               shuffle = True, num_workers = 2)
    test_set = torchvision.datasets.CIFAR10(root = './CIFAR10', train = False,
                                            download = True, transform = transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = 50,
                                              shuffle = False, num_workers = 2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
