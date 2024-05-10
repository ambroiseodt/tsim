"""
Custom torch datasets and dataloaders.
"""

# Author: Ambroise Odonnat <ambroiseodonnattechnologie@gmail.com>
#
# License: MIT
import numpy as np
import torch

from torch.utils.data import DataLoader, Dataset


class EmptyDataset(Dataset):
    r"""An empty Dataset."""

    def __init__(self, float_output=True):
        r"""
        Args:
            float_output (bool): Flag whether to convert the labels to
                                 FloatTensor (``True``) or to LongTensor (``False``).
        """
        self.x = torch.FloatTensor([])
        if float_output:
            self.y = torch.FloatTensor([])
        else:
            self.y = torch.LongTensor([])

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        return self.x, self.y


class LabeledDataset(Dataset):
    r"""A Dataset for labeled data."""

    def __init__(self, x: np.array, y: np.array, float_output=False):
        r"""
        Args:
            x (np.array): Input data. Shape = (n_samples, dimension).
            y (np.array): Corresponding labels. Shape = (n_samples,).
            float_output (bool): Flag whether to convert the labels to
                                 FloatTensor (``True``) or to LongTensor (``False``).
        """
        self.x = torch.FloatTensor(x)

        if float_output:
            self.y = torch.FloatTensor(y)
        else:
            self.y = torch.LongTensor(y)

    def transform(self, x: np.array):
        r"""Obtain input in FloatTensor."""
        return torch.FloatTensor(x)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        examples = self.x[idx]
        labels = self.y[idx]
        return examples, labels


class UnlabeledDataset(Dataset):
    r"""A Dataset for unlabeled data."""

    def __init__(self, x: np.array):
        r"""
        Args:
            x (np.array): Input data. Shape = (n_samples, dimension).
        """

        self.x = torch.FloatTensor(x)

    def transform(self, x: np.array):
        r"""Obtain input in FloatTensor."""
        return torch.FloatTensor(x)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        examples = self.x[idx]
        return examples


class ForeverDataIterator:
    r"""A data iterator that will never stop producing data."""

    def __init__(self, data_loader: DataLoader, device=None):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)
        self.device = device

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data

    def __len__(self):
        return len(self.data_loader)
