# $\mathcal{T}$-similarity
**This is the official implementation of [Leveraging Ensemble Diversity for Robust Self-Training in the Presence of Sample Selection Bias](https://arxiv.org/pdf/2310.14814), AISTATS 2024.**

## Overview
This package implements the **$\mathcal{T}$-similarity**, a drop-in replacement of the softmax for confidence estimation under distribution shifts. This novel confidence measure is build upon a diverse ensemble of linear classifiers and corrects the softmax overconfidence while being calibrated. It can be use for confidence estimation and SSL methods using neural networks as backbone, e.g., self-training. 

## What is included?
More specifically, we provide:
- Labeling procedure to induce distribution shift between labeled and unlabeled data
- Calibrated confidence measure
- Trainer and loss to learn diverse ensembles

### Labling procedure
Sample selection bias (SSB) occurs when data labeling is subject to constraints resulting in a distribution mismatch between labeled and unlabeled data.
We illustrate below the two types of labeling considered in our [paper](https://arxiv.org/pdf/2310.14814):
- **IID**: The usual uniform labeling that verifies the i.i.d. assumption;
- **SSB**: Distribution shift between labeled and unlabeled data.

<p align="center">
  <img src="https://github.com/ambroiseodt/tsim/assets/64415312/ebb2980d-b4e8-49b2-8f45-b8e1be8cea1c" width="400">
</p>

### $\mathcal{T}$-similarity and diversity loss
- Implementation of the $\mathcal{T}$-similarity (nn.Module);
- Implementation of the diversity loss as Pytorch loss (nn.Module).

### Architecture for self-training with the $\mathcal{T}$-similarity 
To combine prediction and confidence estimation, e.g., for self-training, we provide the lightweight architecture below.
        
<p align="center">
  <img src="https://github.com/ambroiseodt/tsim/assets/64415312/797cdfff-0621-420f-bc65-100a50f140cb" width="400">
</p>

#### *Key features*
- Backpropagation of the diversity loss only influences the ensemble, not the projection layers;
- In practice, we use $M=5$ heads resulting in lightweight and fast training;
- Compatible to any SSL methods using neural networks as backbones.

### Notebooks to reproduce the figure from the [paper](https://arxiv.org/pdf/2310.14814)
- Overview of the method (Figure 1)
- Visualization of the sample selection bias (Figure 3)
- $\mathcal{T}$-similarity corrects overconfidence of the softmax (Figure 6)

## What will be added?
- Visualization of ECE for softmax and $\mathcal{T}$-similarity (Figure 5)
- Self-training algorithms
- Clean requirements.txt

## Modules
SAMformer consists of several key modules:
- `models/`: Contains the SAMformer architecture along with necessary components for normalization and optimization.
- `utils/`: Contains the utilities for data processing, training, callbacks, and to save the results.
- `dataset/`: Directory for storing the datasets used in experiments. For illustration purposes, this directory only contains the ETTh1 dataset in .csv format. 

## Installation
To get started with SAMformer, clone this repository and install the required packages (requirements.txt coming soon).

```bash
git clone https://github.com/ambroiseodt/tsim.git
cd tsim
pip install -r requirements.txt
```
Make sure you have Python 3.8 or a newer version installed.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Authors
[Ambroise Odonnat](https://ambroiseodt.github.io/) 

## Note
For the MSLA baseline, there is one difference between this implementation and our paper: the supervised loss $\ell_\mathrm{sup}$ of Eq.(1) is obtained by a single average over the labeled training set instead of averaging separately over the originally labeled data and the pseudo-labeled data. We find that it improves the results, accentuating the superiority of the $\mathcal{T}$-similarity over the $\texttt{softmax}$. For the other baselines, the implementation has not changed (single average over the labeled training set).

## Cite us
If you use our code in your research,  please cite:

```
@InProceedings{pmlr-v238-odonnat24a,
  title = 	 { Leveraging Ensemble Diversity for Robust Self-Training in the Presence of Sample Selection Bias },
  author =       {Odonnat, Ambroise and Feofanov, Vasilii and Redko, Ievgen},
  booktitle = 	 {Proceedings of The 27th International Conference on Artificial Intelligence and Statistics},
  pages = 	 {595--603},
  year = 	 {2024},
  editor = 	 {Dasgupta, Sanjoy and Mandt, Stephan and Li, Yingzhen},
  volume = 	 {238},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {02--04 May},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v238/odonnat24a/odonnat24a.pdf},
  url = 	 {https://proceedings.mlr.press/v238/odonnat24a.html},
}
```



