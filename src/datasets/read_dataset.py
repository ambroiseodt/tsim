"""
Recover data and apply the labeling procedure.
"""

# Author: Ambroise Odonnat <ambroiseodonnattechnologie@gmail.com>
#
# License: MIT

import gzip
import os

import numpy as np

from scipy.io import loadmat
from sklearn.datasets import load_svmlight_file, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .sample_selection_bias import labeled_unlabeled_split


class RealDataSet:
    r"""A class to recover the data and apply the labeling procedure.

    Attributes:
        dataset_name (str): Name of the dataset.
        path_data (str): Path to the data.
        x (np.array): Input data. Shape = (n_samples, dimension).
        y (np.array): Corresponding input labels. Shape = (n_samples,).
        x_train (np.array): Training data. Shape = (n_train, dimension).
        x_test (np.array): Test data. Shape = (n_test, dimension).
        y_train (np.array): Training labels. Shape = (n_train,).
        y_test (np.array): Test labels. Shape = (n_test,).
        labeled_idxes (list): Indexes of training data to include in labeled set.
                            List of length n_l = 100 * lab_size * n_train.
        unlabeled_idxes (list): Indexes of training data to keep unlabeled.
                                List of length n_u = n_train - n_l.
        seed (int): Seed for reproducibility.

    Methods:
        _read_data(): Method to recover the data and labels in arrays.
        get_split(test_size, lab_size, selection_bias): Splits the data in labeled and
                                                        unlabeled training  sets, and test set.
    """

    def __init__(
        self,
        dataset_name: str,
        seed=None,
    ):
        r"""
        Args:
            dataset_name (str): Name of the dataset.
            seed (int): Seed for reproducibility.
        """

        self.dataset_name = dataset_name
        self.path_data = "../data"
        self._read_data()
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.labeled_idxes = None
        self.unlabeled_idxes = None
        self.seed = seed

    def _read_data(self):
        r"""Method to recover the data and labels in arrays."""
        dataset_name = self.dataset_name
        if dataset_name == "codrna":
            self.x, self.y, *_ = load_svmlight_file(
                os.path.join(self.path_data, "codrna.txt")
            )
            self.x = self.x.toarray()
        elif dataset_name == "coil20":
            mat = loadmat(os.path.join(self.path_data, "coil20/coil20.mat"))
            self.x, self.y = mat["X"], mat["Y"].ravel()
        elif dataset_name == "digits":
            digits = load_digits()
            self.x, self.y = digits["data"], digits["target"]
        elif dataset_name == "dna":
            self.x, self.y = _read_gz_dataset(self.path_data, "dna")
        elif dataset_name == "dry_bean":
            self.x, self.y = _read_gz_dataset(self.path_data, "dry_bean")
        elif dataset_name == "har":
            self.x, self.y = _read_gz_dataset(self.path_data, "har")
        elif dataset_name == "mnist":
            loader = MNISTLoader(path_data=self.path_data)
            x = loader.images
            x = x.reshape((x.shape[0], -1)).astype(float)
            y = loader.labels
            self.x, self.y = x, y
        elif dataset_name == "mushrooms":
            self.x, self.y, *_ = load_svmlight_file(
                os.path.join(self.path_data, "mushrooms.txt")
            )
            self.x = self.x.toarray()
        elif dataset_name == "protein":
            self.x, self.y = _read_gz_dataset(self.path_data, "protein")
        if dataset_name == "phishing":
            self.x, self.y, *_ = load_svmlight_file(
                os.path.join(self.path_data, "phishing.txt")
            )
            self.x = self.x.toarray()
        elif dataset_name == "rice":
            self.x, self.y = _read_gz_dataset(self.path_data, "rice")
        if dataset_name == "splice":
            self.x, self.y, *_ = load_svmlight_file(
                os.path.join(self.path_data, "splice.txt")
            )
            self.x = self.x.toarray()
        if dataset_name == "svmguide1":
            self.x, self.y, *_ = load_svmlight_file(
                os.path.join(self.path_data, "svmguide1.txt")
            )
            self.x = self.x.toarray()
        self.x, self.y = _format(self.x, self.y, np.unique(self.y))

    def get_split(
        self,
        test_size: int,
        lab_size: float,
        selection_bias=False,
    ):
        r"""Split the data between labeled and unlabeled training set, and test set.

        The sample selection bias is applied at this level.

        Args:
            test_size (int): Number of samples in the test set.
                            Should be between 0 and n_samples.
            lab_size (float): Proportion of training data to label.
                            Should be between 0.0 and 1.0.
            selection_bias (bool): Flag whether to apply the SSB labeling procedure (``True``)
                                that model sample selection bias or apply the IID labeling
                                procedure (``False``) that verifies the i.d.d. assumption.

        Returns:
            x_l (np.array): Labeled training data. Shape = (n_l, dimension).
            x_u (np.array): Unlabeled training data. Shape = (n_u, dimension).
            y_l (np.array): Labels of training labeled data. Shape = (n_l,).
            y_u (np.array): Labels of training unlabeled data. Shape = (n_u,).
                            Not used for training, only to estimate the transductive error.
            self.x_test (np.array): Test data. Shape = (n_test, dimension).
            self.y_test (np.array): Corresponding test labels. Shape = (n_test).
            n_classes (int): Number of classes.
        """

        n_classes = len(list(set(self.y)))
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=test_size, random_state=self.seed, stratify=self.y
        )

        # Flatten data features
        shape_train = self.x_train.shape
        self.x_train = self.x_train.reshape(shape_train[0], -1)
        shape_test = self.x_test.shape
        self.x_test = self.x_test.reshape(shape_test[0], -1)

        # Standardization
        scaler = StandardScaler()
        self.x_train = scaler.fit_transform(self.x_train)
        self.x_test = scaler.transform(self.x_test)

        # Retrieve original shape
        self.x_train = self.x_train.reshape(*shape_train)
        self.x_test = self.x_test.reshape(*shape_test)

        self.labeled_idxes, self.unlabeled_idxes = labeled_unlabeled_split(
            dataset_name=self.dataset_name,
            x=self.x_train,
            y=self.y_train,
            lab_size=lab_size,
            seed=self.seed,
            selection_bias=selection_bias,
        )
        x_l, y_l = self.x_train[self.labeled_idxes], self.y_train[self.labeled_idxes]
        x_u, y_u = (
            self.x_train[self.unlabeled_idxes],
            self.y_train[self.unlabeled_idxes],
        )

        return (
            x_l,
            x_u,
            y_l,
            y_u,
            self.x_test,
            self.y_test,
            n_classes,
        )


class MNISTLoader:
    r"""A class to load MNIST data.

    Attributes:
        images (np.array): Data. Shape = (n_samples, 28, 28).
        labels (np.array): Corresponding labels. Shape = (n_samples,).

    Methods:
        _extract_images(source): Recover input data from source.
        _extract_labels(source): Recover corresponding labels.
    """

    def __init__(self, path_data="../data"):
        r"""
        Args:
        ----------
        path_data (str): Path to data.
        """

        self.base_path = os.path.join(path_data, "MNIST/raw/")
        train_images = self._extract_images(source="train")
        test_images = self._extract_images(source="t10k")
        images = np.concatenate((train_images, test_images))
        train_labels = self._extract_labels(source="train")
        test_labels = self._extract_labels(source="t10k")
        labels = np.concatenate((train_labels, test_labels))
        self.images = images
        self.labels = labels

    def _extract_images(self, source: str):
        r"""Recover data from source.

        Args:
            source (str): If source is "train", recover training data.
                        If source is "t10k", recover test data.

        Returns:
            images (array): Input data. Shape = (n_samples, 28, 28).
        """

        with gzip.open(self.base_path + source + "-images-idx3-ubyte.gz", "r") as f:
            # First 4 bytes is a magic number
            _ = int.from_bytes(f.read(4), "big")

            # Second 4 bytes is the number of images
            image_count = int.from_bytes(f.read(4), "big")

            # Third 4 bytes is the row count
            row_count = int.from_bytes(f.read(4), "big")

            # Fourth 4 bytes is the column count
            column_count = int.from_bytes(f.read(4), "big")

            # The rest is the image pixel data, each pixel is stored as an unsigned byte
            # pixel values are 0 to 255
            image_data = f.read()
            images = np.frombuffer(image_data, dtype=np.uint8).reshape(
                (image_count, row_count, column_count)
            )
            return images

    def _extract_labels(self, source: str):
        r"""Recover labels from source.

        Args:
            source (str): If source is "train", recover training labels.
                      If source is "t10k", recover test labels.

        Returns:
            labels (array): Input labels. Shape = (n_samples,).
        """

        with gzip.open(self.base_path + source + "-labels-idx1-ubyte.gz", "r") as f:
            # First 4 bytes is a magic number
            _ = int.from_bytes(f.read(4), "big")

            # Second 4 bytes is the number of labels
            _ = int.from_bytes(f.read(4), "big")

            # The rest is the label data, each label is stored as unsigned byte
            # label values are 0 to 9
            label_data = f.read()
            labels = np.frombuffer(label_data, dtype=np.uint8)

            return labels


def _read_gz_dataset(path_data: str, dataset_name: str):
    r"""Recover data and labels.

    Args:
        dataset_name (str): Name of the dataset.
        path_data (str): Path to the data.

    Returns:
        x (np.array): Input data. Shape = (n_samples, dimension).
        y (np.array): Corresponding input labels. Shape = (n_samples,).
    """

    x = np.loadtxt(os.path.join(path_data, dataset_name, dataset_name + "-x.gz"))
    y = np.loadtxt(os.path.join(path_data, dataset_name, dataset_name + "-y.gz"))

    return x, y


def _format(x: np.array, y: np.array, classes: np.array):
    r"""Format data with labels equal to [0, ..., n_classes-1].

    Args:
        x (np.array): Input data. Shape = (n_samples, dimension).
        y (np.array): Corresponding input labels. Shape = (n_samples,).
        classes (np.array): Original set of labels. Shape = (n_classes,).

    Returns:
        x (np.array): Input data. Shape = (n_samples, dimension).
        y (np.array): Corresponding input labels in format [0, ..., n_classes-1].
                      Shape = (n_samples,).
    """

    x = np.asarray(x)
    new_classes = np.arange(len(classes))
    y_new = np.copy(y).astype(int)
    for i, k in enumerate(classes):
        y_new[y == k] = new_classes[i]
    y = np.array(y_new)
    return x, y
