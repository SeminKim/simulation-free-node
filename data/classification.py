import os
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class SVHNWrapper(datasets.SVHN):
    '''
    Simple wrapper to handle split argument
    '''

    def __init__(self, *args, train=True, **kwargs):
        split = 'train' if train else 'test'
        super().__init__(*args, split=split, **kwargs)


# Constants
DATASET_CLASSES = {
    'mnist': datasets.MNIST,
    'cifar10': datasets.CIFAR10,
    'cifar100': datasets.CIFAR100,
    'svhn': SVHNWrapper,
}

DATASET_NUM_CLASSES = {
    'mnist': 10,
    'cifar10': 10,
    'cifar100': 100,
    'svhn': 10,
}

DATASET_IMG_SIZE = {
    'mnist': 28,
    'cifar10': 32,
    'cifar100': 32,
    'svhn': 32,
}

DATASET_PATHNAME = {
    'mnist': 'mnist',
    'cifar10': 'cifar',
    'cifar100': 'cifar',
    'svhn': 'svhn',
}


def get_transform(data_aug=True, dataset='mnist', aug_type='basic'):
    '''
    Get the data augmentation and normalization transformations for the dataset
    '''
    if dataset == 'mnist':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        if data_aug:
            transform_train = transforms.Compose([
                transforms.RandomCrop(28, padding=4),
                transforms.ToTensor(),
            ])
        else:
            transform_train = transform_test

    elif dataset in ['cifar10', 'cifar100']:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        if data_aug:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            transform_train = transform_test

    elif dataset == 'svhn':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])
        if data_aug:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
            ])
        else:
            transform_train = transform_test

    return transform_train, transform_test


def to_one_hot(num_classes):
    '''
    Convert labels to one-hot encoding
    '''
    def to_one_hot_fn(x):
        return torch.nn.functional.one_hot(torch.tensor(x), num_classes).float()
    return to_one_hot_fn


def get_datasets(root='.data', name='mnist', data_aug=True, perc=None, stride=None, onehot=True, verbose=True, aug_type='basic'):
    # download flag
    base_path = os.path.join(root, DATASET_PATHNAME[name])
    download = not os.path.exists(base_path)
    # stride for train subset evaluation
    assert perc is None or stride is None, 'Only one of perc or stride can be set'
    if perc is not None:
        stride = int(1 / perc)
    if stride is None:
        stride = 1
    # get transforms
    transform_train, transform_test = get_transform(data_aug, name, aug_type)
    num_classes = DATASET_NUM_CLASSES[name]
    target_transform = to_one_hot(num_classes) if onehot else None

    if name in ['mnist', 'cifar10', 'cifar100', 'svhn']:
        dataset_class = DATASET_CLASSES[name]
        train_dataset = dataset_class(root=base_path, train=True, download=download, transform=transform_train,
                                      target_transform=target_transform)
        eval_dataset = dataset_class(root=base_path, train=True, download=download, transform=transform_test,
                                     target_transform=target_transform)
        eval_dataset = torch.utils.data.Subset(eval_dataset, list(range(0, len(eval_dataset), stride)))
        test_dataset = dataset_class(root=base_path, train=False, download=download, transform=transform_test,
                                     target_transform=target_transform)

    if verbose:
        print(f'Initialized {name} dataset with {len(train_dataset)} training samples, '
              f'{len(eval_dataset)} evaluation samples, and {len(test_dataset)} test samples')

    return train_dataset, eval_dataset, test_dataset


def get_dataloaders(train_set, val_set, test_set, batch_size=128, test_batch_size=1000, nw=4, **kwargs):
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=nw, drop_last=True, pin_memory=True,
        persistent_workers=True, **kwargs,
    )
    val_loader = DataLoader(
        val_set, batch_size=test_batch_size, shuffle=False, num_workers=nw, drop_last=False, pin_memory=True,
        persistent_workers=True, **kwargs,
    )
    test_loader = DataLoader(
        test_set, batch_size=test_batch_size, shuffle=False, num_workers=nw, drop_last=False, pin_memory=True,
        persistent_workers=True, **kwargs,
    )

    return train_loader, val_loader, test_loader
