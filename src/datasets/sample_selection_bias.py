"""
Labeling procedure to split between labeled and unlabeled
training sets, and test set.
"""

# Author: Ambroise Odonnat <ambroiseodonnattechnologie@gmail.com>
#
# License: MIT
import numpy as np

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def labeled_unlabeled_split(
    dataset_name: str,
    x: np.array,
    y: np.array,
    lab_size: float,
    selection_bias: bool,
    seed: int,
):
    r"""Apply the labeling procedure on training data.

    If selection_bias is True, apply sample selection bias.
    Otherwise, labeled data are drawn uniformly so that the classes are balanced.

    Args:
        dataset_name (str): Name of the dataset.
        x (np.array): Input training data. Shape = (n_train, dimension).
        y (np.array): Corresponding training labels. Shape = (n_train,).
        lab_size (float): Proportion of training data to label.
                            Should be between 0.0 and 1.0.
        selection_bias (bool): Flag whether to apply the SSB labeling procedure (``True``)
                                that model sample selection bias or apply the IID labeling
                                procedure (``False``) that verifies the i.d.d. assumption.
        seed (int): Seed for reproducibility.

    Returns:
        labeled_idxes (list): Indexes of training data to include in labeled set.
                            List of length n_l = 100 * lab_size * n_train.
        unlabeled_idxes (list): Indexes of training data to keep unlabeled.
                                List of length n_u = n_train - n_l.
    """

    if lab_size == 1.0:
        print("Supervised task: unlabeled set is empty")
        labeled_idxes, unlabeled_idxes = np.arange(len(x)), []
    else:
        if selection_bias:
            if dataset_name == "codrna":
                a = 2
                b = 1
            elif dataset_name == "coil20":
                a = 0.5
                b = 1.5
            elif dataset_name == "digits":
                a = 0.5
                b = 1
            elif dataset_name == "dna":
                a = 50
                b = 2
            elif dataset_name == "dry_bean":
                a = 2
                b = 1
            elif dataset_name == "har":
                a = 0.5
                b = 1.5
            elif dataset_name == "mnist":
                a = 0.5
                b = 1.5
            elif dataset_name == "mushrooms":
                a = 2
                b = 1
            elif dataset_name == "phishing":
                a = 2
                b = 1
            elif dataset_name == "protein":
                a = 0.6
                b = 1
            elif dataset_name == "rice":
                a = 2
                b = 1
            elif dataset_name == "splice":
                a = 2
                b = 1
            elif dataset_name == "svmguide1":
                a = 2
                b = 1
            assert b > 0, "Invalid value of b."
            ratio = a / b
            labeled_idxes, unlabeled_idxes = _pca_split(x, y, lab_size, ratio, seed)
        else:
            labeled_idxes, unlabeled_idxes = _balanced_split(y, lab_size, seed)

    return labeled_idxes, unlabeled_idxes


def _balanced_split(
    y: np.array,
    lab_size: float,
    seed: int,
    shuffle_data=True,
):
    r"""Apply IID labeling procedure on training data.

    Labeled data are drawn uniformly so that classes are balanced.

    Args:
        y (np.array): Training labels. Shape = (n_train,).
        lab_size (float): Proportion of training data to label.
                            Should be between 0.0 and 1.0.
        seed (int): Seed for reproducibility.

    Returns:
        labeled_idxes (list): Indexes of training data to include in labeled set.
                            List of length n_l = 100 * lab_size * n_train.
        unlabeled_idxes (list): Indexes of training data to keep unlabeled.
                                List of length n_u = n_train - n_l.
    """

    # Define random state
    rng = np.random.default_rng(seed)

    # Number of labeled samples per class
    class_labels = np.unique(y)
    lab_samples_per_class = int(lab_size * len(y) / len(class_labels))
    n_labeled = lab_samples_per_class * len(class_labels)
    assert n_labeled <= len(y), "Too many labels selected"

    # Labeled subset
    labeled_idxes = []
    for i in class_labels:
        ith_class_idxes = np.where(y == i)[0]
        ith_class_idxes = rng.choice(
            ith_class_idxes, lab_samples_per_class, replace=False
        )
        labeled_idxes.extend(ith_class_idxes)

    # Unlabeled subset
    unlabeled_idxes = [i for i in range(len(y)) if i not in labeled_idxes]

    if shuffle_data:
        labeled_idxes = rng.choice(labeled_idxes, len(labeled_idxes), replace=False)
        unlabeled_idxes = rng.choice(
            unlabeled_idxes, len(unlabeled_idxes), replace=False
        )

    return labeled_idxes, unlabeled_idxes


def _pca_split(
    x: np.array,
    y: np.array,
    lab_size: float,
    ratio: float,
    seed: int,
    shuffle_data=True,
):
    r"""Apply SSB labeling procedure on training data.

    Labeled data are drawn with a probability that depends on the absolute value of
    their projection on the first principal component (PC1). This is an instance of
    sample selection bias where the distribution of classes is preserved.

    Args:
        x (np.array): Input training data. Shape = (n_train, dimension).
        y (np.array): Corresponding training labels. Shape = (n_train,).
        lab_size (float): Proportion of training data to label.
                            Should be between 0.0 and 1.0.
        ratio (float): Hyperparameter to model the probability of labeling
                    a training sample. It should be positive.
        seed (int): Seed for reproducibility.

    Returns:
        labeled_idxes (list): Indexes of training data to include in labeled set.
                            List of length n_l = 100 * lab_size * n_train.
        unlabeled_idxes (list): Indexes of training data to keep unlabeled.
                                List of length n_u = n_train - n_l.
    """
    assert lab_size < 1, "Too many labels selected"

    # Flatten data features
    shape = x.shape
    x = x.reshape(shape[0], -1)

    # Standardization
    tol = 1e-12
    if (abs(x.mean()) > tol) and (abs(x.std() - 1) > tol):
        scaler = StandardScaler()
        x = scaler.fit_transform(x)

    # Define random state
    rng = np.random.default_rng(seed)

    # Find a number of samples in train/test for each class
    class_labels = np.unique(y)
    y_train_dummy, y_test_dummy = train_test_split(
        y, train_size=lab_size, random_state=seed, stratify=y
    )
    class_distr_train = [np.sum(y_train_dummy == k) for k in class_labels]
    class_distr_test = [np.sum(y_test_dummy == k) for k in class_labels]

    labeled_idxes, unlabeled_idxes = list(), list()
    for i, k in enumerate(class_labels):
        x_k = x[y == k]
        idx_k = np.where(y == k)[0]
        n_train_take = class_distr_train[i]
        n_test_take = class_distr_test[i]

        # PCA
        pca = PCA()
        components = pca.fit_transform(x_k)
        projection = components[:, 0]
        densities = np.exp(ratio * np.abs(projection))

        # Sample selection bias
        sample_bias = densities / densities.sum()
        train_inds = rng.choice(
            np.arange(len(x_k)), n_train_take, p=sample_bias, replace=False
        )
        test_inds = np.setdiff1d(np.arange(n_train_take + n_test_take), train_inds)
        labeled_idxes.extend(idx_k[train_inds])
        unlabeled_idxes.extend(idx_k[test_inds])

    if shuffle_data:
        labeled_idxes = rng.choice(labeled_idxes, len(labeled_idxes), replace=False)
        unlabeled_idxes = rng.choice(
            unlabeled_idxes, len(unlabeled_idxes), replace=False
        )

    return labeled_idxes, unlabeled_idxes
