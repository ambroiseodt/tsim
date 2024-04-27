r"""
Implementation of the T-similarity in Definition 3.1 and the diversity loss in Eq.(2) of [1].

[1] A. Odonnat, V. Feofanov, I. Redko. Leveraging Ensemble Diversity
    for Robust Self-Training in the presence of Sample Selection Bias.
    International Conference on Artifical Intelligence and Statistics (AISTATS), 2024
"""

# Author: Ambroise Odonnat <ambroiseodonnattechnologie@gmail.com>
#
# License: MIT

import torch

from torch import nn


class Tsimilarity(nn.Module):

    def __init__(self, classwise=False):
        super(Tsimilarity, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.classwise = classwise

    def forward(self, *outputs):
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

        return t_similarity / acc


class DiversityLoss(nn.Module):
    def __init__(self):
        super(DiversityLoss, self).__init__()
        self.tsimilarity_function = Tsimilarity(classwise=False)

    def forward(self, *outputs):
        t_similarity = self.tsimilarity_function(*outputs)
        return -t_similarity.mean(axis=-1)
