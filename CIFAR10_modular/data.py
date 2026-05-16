"""Film-roll CIFAR-10 dataset.

Each sample is a horizontal concatenation of N CIFAR-10 images drawn
independently of each other; the target is the length-N vector of their
class labels.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def default_transform(train: bool = True) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


class ConcatCIFAR10(Dataset):
    """CIFAR-10 with each item formed by concatenating N frames width-wise.

    ``__init__`` only loads the underlying CIFAR-10 split (all images and
    labels live in ``self.base``); the N frames that make up a returned
    sample are drawn uniformly at random *inside* ``__getitem__``, so the
    same ``idx`` will generally yield a different concatenation on each
    access. ``torch.randint`` is used for sampling so each DataLoader worker
    is automatically seeded independently.

    Args:
        root: directory in which the CIFAR-10 data lives (or will be downloaded).
        N: number of frames per sample.
        train: use the train split if True, otherwise the test split.
        transform: optional torchvision transform applied per frame before
            concatenation. Defaults to :func:`default_transform`.
        download: download CIFAR-10 if not already present.

    Returns from ``__getitem__``:
        x: tensor of shape ``(3, 32, 32 * N)``.
        y: tensor of shape ``(N,)`` of integer labels in ``[0, 10)``.
    """

    def __init__(
        self,
        root: str,
        N: int,
        train: bool = True,
        transform=None,
        download: bool = True,
    ):
        if N < 1:
            raise ValueError("N must be >= 1")
        self.N = N
        if transform is None:
            transform = default_transform(train)
        self.base = datasets.CIFAR10(
            root=root, train=train, transform=transform, download=download,
        )

    def __len__(self) -> int:
        # Not a meaningful quantity — samples are drawn at random in
        # __getitem__ — but DataLoader needs a length. We define one
        # "epoch" as enough draws to consume as many frames as the
        # underlying split has images.
        return len(self.base) // self.N

    def __getitem__(self, idx: int):
        indices = torch.randint(0, len(self.base), (self.N,)).tolist()
        imgs, labels = [], []
        for j in indices:
            img, lab = self.base[j]
            imgs.append(img)
            labels.append(lab)
        x = torch.cat(imgs, dim=-1)
        y = torch.tensor(labels, dtype=torch.long)
        return x, y
