r"""
Implementation of the T-similarity and the corresponding diversity loss.
"""

# Author: Ambroise Odonnat <ambroiseodonnattechnologie@gmail.com>
#
# License: MIT

import torch

from torch import nn


class Tsimilarity(nn.Module):
    r"""Implementation of the T-similarity following Definition 3.1 of [1].

    Attributes:
        softmax (nn.Module): Softmax function.
        classwise (bool): Flag whether to make the T-similarity multi-dimensional (``True``)
                            or uni-dimensional as in Definition 3.1 of [1] (``False``).
                            We use classwise=True when the pseudo-labeling policy is MSTA [2].

    References
    ----------
    [1] A. Odonnat, V. Feofanov, I. Redko. Leveraging Ensemble Diversity
        for Robust Self-Training in the presence of Sample Selection Bias.
        International Conference on Artifical Intelligence and Statistics (AISTATS), 2024

    [2] V. Feofanov, E. Devijver, M-R. Amini. Multi-class Probabilistic Bounds
        for Majority Vote Classifiers with Partially Labeled Data.
        Journal of Machine Learning Research (JMLR), 2024
    """

    def __init__(self, classwise=False):
        r"""
        Args:
            classwise (bool): Flag whether to make the T-similarity multi-dimensional (``True``)
                              or uni-dimensional as in Definition 3.1 of [1] (``False``).
                              We use classwise=True when the pseudo-labeling policy is MSTA [2].
        References
        ----------
        [1] A. Odonnat, V. Feofanov, I. Redko. Leveraging Ensemble Diversity
            for Robust Self-Training in the presence of Sample Selection Bias.
            International Conference on Artifical Intelligence and Statistics (AISTATS), 2024

        [2] V. Feofanov, E. Devijver, M-R. Amini. Multi-class Probabilistic Bounds
            for Majority Vote Classifiers with Partially Labeled Data.
            Journal of Machine Learning Research (JMLR), 2024
        """
        super(Tsimilarity, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.classwise = classwise

    def forward(self, *outputs):
        r"""
        Args:
            *outputs (unpacked list of torch.Tensor): Unpacked list of the ensemble outputs.
                                                      Each tensor is of shape (batch_size, n_classes).

        Returns:
            t_similarity (torch.Tensor): T-similarity of the ensemble.
                                         If classwise=True, shape = (batch_size, n_classes),
                                         otherwise shape = (batch_size,).
        """
        if self.classwise:
            t_similarity = torch.zeros_like(outputs[0])
        else:
            t_similarity = 0
        n_classifiers = len(outputs)
        acc = 0
        for i in range(n_classifiers):
            for j in [k for k in range(n_classifiers) if k != i]:
                p_i = self.softmax(outputs[i])
                p_j = self.softmax(outputs[j])
                if self.classwise:
                    t_similarity += p_i * p_j
                else:
                    t_similarity += (p_i * p_j).sum(dim=-1)
                acc += 1
        t_similarity /= acc
        return t_similarity


class DiversityLoss(nn.Module):
    r"""Implementation of the diversity loss following Eq.(2) of [1].

    Attributes:
        tsimilarity_function (nn.Module): T-similarity function.

    References
    ----------
    [1] A. Odonnat, V. Feofanov, I. Redko. Leveraging Ensemble Diversity
        for Robust Self-Training in the presence of Sample Selection Bias.
        International Conference on Artifical Intelligence and Statistics (AISTATS), 2024
    """

    def __init__(self):
        r"""
        Args:
            tsimilarity_function (nn.Module): T-similarity function.
        """
        super(DiversityLoss, self).__init__()
        self.tsimilarity_function = Tsimilarity(classwise=False)

    def forward(self, *outputs):
        r"""
        Args:
            *outputs (unpacked list of torch.Tensor): Unpacked list of the ensemble's outputs.
                                                      Each tensor is of shape (batch_size, n_classes).

        Returns:
            diversity_loss (float): Diversity loss of the ensemble.
        """
        t_similarity = self.tsimilarity_function(*outputs)
        diversity_loss = -t_similarity.mean(axis=-1)

        return diversity_loss
