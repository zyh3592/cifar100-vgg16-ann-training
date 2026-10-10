import runtime
import random
import numpy as np
import torch
from torchvision import datasets, transforms

MEAN = (0.5071, 0.4867, 0.4408)
STD = (0.2675, 0.2565, 0.2761)


def transform(train=False, image_augmentation='none'):
    steps = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] if train else []
    if image_augmentation not in {'none', 'autoaugment'}:
        raise ValueError('Unknown image augmentation')
    if train and image_augmentation == 'autoaugment':
        steps.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10))
    return transforms.Compose(steps + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def stratified_split(targets, val_per_class, seed):
    if not 1 <= val_per_class < 500:
        raise ValueError('val_per_class must be between 1 and 499')
    rng = np.random.default_rng(seed)
    labels = np.asarray(targets)
    training, validation = [], []
    if set(labels.tolist()) != set(range(100)):
        raise ValueError('Expected exactly 100 CIFAR-100 classes')
    for label in range(100):
        indices = np.flatnonzero(labels == label)
        if len(indices) != 500:
            raise ValueError('Expected 500 training images per class')
        rng.shuffle(indices)
        validation.extend(indices[:val_per_class].tolist())
        training.extend(indices[val_per_class:].tolist())
    return training, validation


def training_datasets(root, val_per_class=50, seed=42, image_augmentation='none'):
    augmented = datasets.CIFAR100(root=str(root), train=True,
                                 transform=transform(True, image_augmentation), download=False)
    clean = datasets.CIFAR100(root=str(root), train=True, transform=transform(False), download=False)
    train_indices, val_indices = stratified_split(augmented.targets, val_per_class, seed)
    return (torch.utils.data.Subset(augmented, train_indices),
            torch.utils.data.Subset(clean, val_indices),
            augmented.classes, train_indices, val_indices)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
