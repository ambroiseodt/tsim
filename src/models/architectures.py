r"""
Neural networks architectures and ensemble classifier.
"""

# Author: Ambroise Odonnat <ambroiseodonnattechnologie@gmail.com>
#
# License: MIT

import numpy as np

import torch
from torch import nn


class MLP(nn.Module):
    r"""Multi-Layer Perceptron whose hidden dimensions depends on the input's shape."""

    def __init__(self, input_shape: tuple):
        r"""
        Args:
            input_shape (tuple): Dimension of the input. For MNIST data, we have:
                                    - input_shape = (784, ) if inputs are flattened;
                                    - input_shape = (1, 28, 28) otherwise.
        """

        super().__init__()
        input_dim = np.array(input_shape).prod()

        # Get feature dimension
        self.features = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, input_dim // 8),
            nn.ReLU(),
        )
        self._out_features = input_dim // 8

    def forward(self, x: torch.Tensor):
        r"""
        Flatten the inputs and apply the neural network.

        Args:
            x (torch.Tensor): Input tensor. Shape = (batch_size, dimension).

        Returns:
            x (torch.Tensor): Neural network's embedding to be fed to the classifier.
                              Shape = (batch_size, hidden_dim).
        """

        x = torch.flatten(x, 1)
        x = self.features(x)
        return x

    @property
    def out_features(self) -> int:
        r"""Return the dimension of the neural network's embedding."""
        return self._out_features


class MLPfixed(nn.Module):
    r"""Multi-Layer Perceptron whose hidden dimensions are fixed."""

    def __init__(self, input_shape: tuple):
        r"""
        Args:
            input_shape (tuple): Dimension of the input. For MNIST data, we have:
                                    - input_shape = (784, ) if inputs are flattened;
                                    - input_shape = (1, 28, 28) otherwise.
        """
        super().__init__()
        input_dim = np.array(input_shape).prod()

        # Get feature dimension
        self._out_features = 32
        self.features = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self._out_features),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor):
        r"""
        Flatten the inputs and apply the neural network.

        Args:
            x (torch.Tensor): Input tensor. Shape = (batch_size, dimension).

        Returns:
            out (torch.Tensor): Neural network's embedding to be fed to the classifier.
                              Shape = (batch_size, hidden_dim).
        """
        x = torch.flatten(x, 1)
        out = self.features(x)

        return out

    @property
    def out_features(self) -> int:
        r"""Return the dimension of the neural network's embedding."""
        return self._out_features


class EnsembleClassifier(nn.Module):
    r"""Ensemble of classifiers.

    Attributes:
        n_classifiers (int): Number of classifiers.
        n_classes (int): Number of classes.
        backbone (nn.Module): Neural network backbone
        features_dim (int): Feature dimension.
        pred_head (nn.Module): Prediction classifier.
        heads (nn.ModuleList): list of classifiers.
    """

    def __init__(
        self, input_shape: tuple, n_classes: int, n_classifiers=5, backbone=None
    ):
        r"""

        Args:
            input_shape (tuple): Dimension of the input. For MNIST data, we have:
                                    - input_shape = (784, ) if inputs are flattened;
                                    - input_shape = (1, 28, 28) otherwise.
            n_classes (int): Number of classes.
            n_classifiers (int): Number of classifiers. Defaults to 5.
            backbone (optional, nn.Module): Backbone neural network to obtain feature embeddings.
                                            Defaults to None.
        """
        super().__init__()
        self.n_classifiers = n_classifiers
        self.n_classes = n_classes

        # Feature extractor
        if backbone is not None:
            self.backbone = backbone
        else:
            input_dim = np.array(input_shape).prod()
            self.backbone = (
                MLP(input_shape) if input_dim > 50 else MLPfixed(input_shape)
            )

        self.features_dim = self.backbone.out_features

        # Prediction head
        self.pred_head = nn.Linear(self.features_dim, self.n_classes)

        # Ensemble heads
        self.heads = nn.ModuleList(
            [nn.Linear(self.features_dim, self.n_classes) for i in range(n_classifiers)]
        )

    def forward(self, x: torch.Tensor):
        r"""
        Return neural network prediction of the ensemble heads and the classification head.

        Args:
            x (torch.Tensor): Input tensor. Shape = (batch_size, dimension).

        Returns:
            output_pred_head, outputs_ensemble_heads (Tuple):
                - output_pred_head (torch.Tensor): Output of the prediction head.
                - outputs_ensemble_heads (list of torch.Tensors): Outputs of the ensemble heads.
        """
        f = self.backbone(x)
        f_clone = f.detach().clone()
        output_pred_head = self.pred_head(f)
        outputs_ensemble_heads = [
            self.heads[i](f_clone) for i in range(self.n_classifiers)
        ]

        return output_pred_head, outputs_ensemble_heads
