r"""
Trainer to obtain diverse ensembles in the style of sklearn base_estimator.

[1] A. Odonnat, V. Feofanov, I. Redko. Leveraging Ensemble Diversity
     for Robust Self-Training in the presence of Sample Selection Bias.
     International Conference on Artifical Intelligence and Statistics (AISTATS), 2024
"""

# Author: Ambroise Odonnat <ambroiseodonnattechnologie@gmail.com>
#
# License: MIT

import random
import torch
import numpy as np

from scipy.special import softmax

from torch import nn
from torch.utils.data import DataLoader

from .architectures import EnsembleClassifier
from .tsim import Tsimilarity, ClasswiseTsimilarity, DiversityLoss
from .utils import EmptyDataset, LabeledDataset, UnlabeledDataset, ForeverDataIterator


class DiverseEnsembleMLP:
    r"""A class to fit and predict with a prediction head and diverse ensemble.

    Attributes:
        dataset_l (LabeledDataset): Labeled training set.
        dataset_u (UnlabeledDataset): Unlabeled training set.
        n_classifiers (int): Number of classifiers in the ensemble.
        num_epochs (int): Number of epochs.
        learning_rate (float): Learning rate.
        weight_decay (float): Weight_decay.
        gamma (float): Parameter that controls the diversity strength.
        batch_size_l (int): Batch size for the labeled training set.
        batch_size_u (int): Batch size for the unlabeled training set.
        n_iters (int): Number of iterations in each epoch.
        n_jobs (int): Number of threads used for intraop parallelism on CPU.
        device (torch.device): Device. Defaults="cpu".
        random_state (int): Seed for reproducibility.
        verbose (bool): Flag whether to print the evolution of the training.
        network (nn.Module): Base neural network.
        optimizer (torch.optim.Optimizer): Optimizer.
        supervised_loss (nn.Module): Loss for the supervised prediction head.
        diversity_loss (nn.Module): Diversity loss for the ensemble.
        t_similarity_function (nn.Module): T-similarity.
        classwise_t_similarity_function (nn.Module): Classwise T-similarity.

    Methods:
        fit(x_l, y_l, x_u): Train prediction head on labeled data and
                            ensemble on labeled and unlabeled data.
        predict(x): Predict label of input x using the supervised prediction head.
        predict_proba(x): Predict softmax probabilities of the input x using the
                          supervised prediction head.
        predict_t_similarity(x, classwise): Predict the t_similarity of the input x
                                            using the diverse ensemble. Classwise is
                                            a flag whether to use the classwise
                                            version of T-similarity (``True``) or the
                                            original T-similarity (``False``).
    """

    def __init__(
        self,
        n_classifiers=5,
        num_epochs=5,
        learning_rate=1e-3,
        weight_decay=0,
        n_iters=100,
        gamma=1,
        batch_size_l=32,
        batch_size_u=32,
        n_jobs=1,
        device="cpu",
        random_state=None,
        verbose=True,
    ):
        """
        Args:
            n_classifiers (int): Number of classifiers in the ensemble.
            num_epochs (int): Number of epochs.
            learning_rate (float): Learning rate.
            n_iters (int): Number of iterations in each epoch.
            weight_decay (float): Weight_decay.
            gamma (float): Parameter that controls the diversity strength.
            batch_size_l (int): Batch size for the labeled training set.
            batch_size_u (int): Batch size for the unlabeled training set.
            n_jobs (int): Number of threads used for intraop parallelism on CPU.
            device (torch.device): Device. Defaults="cpu".
            random_state (int): Seed for reproducibility.
            verbose (bool): Flag whether to print the evolution of the training.
        """
        self.dataset_l = LabeledDataset
        self.dataset_u = UnlabeledDataset
        self.n_classifiers = n_classifiers
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.batch_size_l = batch_size_l
        self.batch_size_u = batch_size_u
        self.n_iters = n_iters
        self.n_jobs = n_jobs
        self.device = device
        self.random_state = random_state
        self.verbose = verbose
        self.network = None
        self.optimizer = None
        self.supervised_loss = nn.CrossEntropyLoss()
        self.diversity_loss = DiversityLoss()
        self.t_similarity_function = Tsimilarity().to(self.device)
        self.classwise_t_similarity_function = ClasswiseTsimilarity().to(self.device)

    def fit(self, x_l: np.array, y_l: np.array, x_u: np.array):
        """Train the prediction head with labeled data and the ensemble with labeled
        and unlabeled data to promote diverstiy.
        Args:
            x_l (np.array): Labeled training data. Shape = (n_l, dimension).
            x_l (np.array): Corresponding labels. Shape = (n_l, dimension).
            x_l (np.array): Unlabeled training data. Shape = (n_u, dimension).
        """
        # Set seed
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            random.seed(self.random_state)

        torch.set_num_threads(self.n_jobs)

        # Model
        network = EnsembleClassifier(
            input_shape=x_l.shape[1:],
            n_classes=np.unique(y_l).size,
            n_classifiers=self.n_classifiers,
        )
        if self.device == "cpu":
            self.network = network.to(self.device)
        else:
            self.network = nn.DataParallel(network).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Loss
        self.supervised_loss = self.supervised_loss.to(self.device)
        self.diversity_loss = self.diversity_loss.to(self.device)

        # Dataloader
        self.batch_size_l = min(x_l.shape[0], self.batch_size_l)
        self.batch_size_u = min(x_u.shape[0], self.batch_size_u)
        train_l = self.dataset_l(x_l, y_l)
        if self.batch_size_u == 0:
            train_u = EmptyDataset(float_output=False)
        else:
            train_u = self.dataset_u(x_u)
        dataloader_l = DataLoader(train_l, batch_size=self.batch_size_l, num_workers=0)
        dataloader_u = DataLoader(train_u, num_workers=0)
        iterator_l, iterator_u = ForeverDataIterator(dataloader_l), ForeverDataIterator(
            dataloader_u
        )

        # Training
        self.network = self.network.train()
        for epoch in range(self.num_epochs):
            for _ in range(self.n_iters):

                # Batches of labeled and unlabeled data
                x_l_batch, y_l_batch = next(iterator_l)
                x_l_batch, y_l_batch = (
                    x_l_batch.to(self.device),
                    y_l_batch.to(self.device),
                )
                if self.batch_size_u == 0:
                    x_u_batch = torch.FloatTensor([])
                else:
                    x_u_batch = next(iterator_u)
                x_u_batch = x_u_batch.to(self.device)

                # ===================Forward====================
                output_pred_head, outputs_ensemble_heads = self.network(
                    torch.cat([x_l_batch, x_u_batch], axis=0),
                )
                outputs_l = [
                    output[: x_l_batch.shape[0]] for output in outputs_ensemble_heads
                ]
                outputs_u = [
                    output[x_l_batch.shape[0] :] for output in outputs_ensemble_heads
                ]

                # Supervised loss
                loss = self.supervised_loss(
                    output_pred_head[: x_l_batch.shape[0]], y_l_batch
                )

                # Ensemble supervised loss
                for i in range(self.n_classifiers):
                    loss += (
                        self.supervised_loss(outputs_l[i], y_l_batch)
                        / self.n_classifiers
                    )

                # Ensemble diversity loss
                # We add "- gamma * diversity_term" because we want to maximize the diversity
                if self.gamma != 0:
                    diversity_term = self.diversity_loss(*outputs_u).mean()
                    loss -= self.gamma * diversity_term

                # ===================Backward====================
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if self.verbose:
                print(f"Epoch [{epoch + 1}/{self.num_epochs}]")
                print(f"Loss:{loss:.4f}, Diversity:{diversity_term:.4f}")

    def predict(self, x: np.array):
        """Predict with the supervised prediction head label of an input x.

        Args:
            x (np.array): Input data. Shape = (n_samples, dimension).

        Returns:
            predicted_label (np.array): Predicted label. Shape = (n_samples,)
        """
        x = torch.FloatTensor(x).to(self.device)
        with torch.no_grad():
            output_pred_head, _ = self.network(x)
            out = output_pred_head.cpu().numpy()
        predicted_label = out.argmax(axis=-1)

        return predicted_label

    def predict_proba(self, x: np.array):
        """Predict with the supervised prediction head the probability
        distribution on the classes for an input x.

        Args:
            x (np.array): Input data. Shape = (n_samples, dimension).

        Returns:
            predicted_proba (np.array): Predicted probability distribution on the classes.
                                        Shape = (n_samples, n_classes)
        """
        x = torch.FloatTensor(x).to(self.device)
        with torch.no_grad():
            output_pred_head, _ = self.network(x)
            out = output_pred_head.cpu().numpy()
        predicted_proba = softmax(out, axis=-1)

        return predicted_proba

    def predict_t_similarity(self, x: np.array, classwise=False):
        """Predict T-similarity of an input x with the ensemble heads.

        Args:
            x (np.array): Input data.
            classwise (bool): Flag whether to use the classwise version of the
                              T-similarity (``True``) or the original one (``False``).

        Returns:
            tsim (np.array): T-similarity of an input x.
                             If classwise is False, shape = (n_samples, ).
                             Otherwise, shape = (n_samples, n_classes).
        """
        x = torch.FloatTensor(x).to(self.device)
        if classwise:
            t_similarity_function = self.classwise_t_similarity_function
        else:
            t_similarity_function = self.t_similarity_function
        t_similarity_function = t_similarity_function.to(self.device)
        with torch.no_grad():
            _, outputs_ensemble_heads = self.network(x)
            tsim = t_similarity_function(*outputs_ensemble_heads)
        tsim = tsim.cpu().numpy()

        return tsim
