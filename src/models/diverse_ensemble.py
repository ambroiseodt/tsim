r"""
Trainer to obtain diverse ensembles following Figure 2 of [1].

[1] A. Odonnat, V. Feofanov, I. Redko. Leveraging Ensemble Diversity
     for Robust Self-Training in the presence of Sample Selection Bias.
     International Conference on Artifical Intelligence and Statistics (AISTATS), 2024
"""

# Author: Ambroise Odonnat <ambroiseodonnattechnologie@gmail.com>
#
# License: MIT

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
from .tsim import Tsimilarity, DiversityLoss
from .utils import EmptyDataset, LabeledDataset, UnlabeledDataset, ForeverDataIterator


class DiverseEnsembleMLP:

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
        self.supervised_loss = None
        self.diversity_loss = None

    def fit(self, x_l: np.array, y_l: np.array, x_u: np.array):

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
        self.supervised_loss = nn.CrossEntropyLoss().to(self.device)
        self.diversity_loss = DiversityLoss().to(self.device)

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
        x = torch.FloatTensor(x).to(self.device)
        with torch.no_grad():
            output_pred_head, _ = self.network(x)
            out = output_pred_head.cpu().numpy()
        return out.argmax(axis=-1)

    def predict_proba(self, x: np.array):
        x = torch.FloatTensor(x).to(self.device)
        with torch.no_grad():
            output_pred_head, _ = self.network(x)
            out = output_pred_head.cpu().numpy()
        return softmax(out, axis=-1)

    def predict_t_similarity(self, x: np.array, classwise=False):
        x = torch.FloatTensor(x).to(self.device)
        tsimilarity_function = Tsimilarity(classwise=classwise).to(self.device)
        with torch.no_grad():
            _, outputs_ensemble_heads = self.network(x)
            tsim = tsimilarity_function(*outputs_ensemble_heads)
        return tsim.cpu().numpy()
