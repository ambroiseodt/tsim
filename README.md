# $\mathcal{T}$-similarity
**This is the official implementation of [Leveraging Ensemble Diversity for Robust Self-Training in the Presence of Sample Selection Bias](https://arxiv.org/pdf/2310.14814), AISTATS 2024.**

## Overview
This package implements the **$\mathcal{T}$-similarity**, a drop-in replacement of the softmax for confidence estimation under distribution shifts. This novel confidence measure is build upon a diverse ensemble of linear classifiers and corrects the softmax overconfidence while being calibrated. It can be use for confidence estimation and SSL methods using neural networks as backbone, e.g., self-training.

## What is included?
We provide the implementation of the following methods.

### Labeling procedure
Sample selection bias (SSB) occurs when data labeling is subject to constraints resulting in a distribution mismatch between labeled and unlabeled data.
We illustrate below the two types of labeling considered in our [paper](https://arxiv.org/pdf/2310.14814):
- **IID**: The usual uniform labeling that verifies the i.i.d. assumption;
- **SSB**: Distribution shift between labeled and unlabeled data.

<p align="center">
  <img src="https://github.com/ambroiseodt/tsim/assets/64415312/ebb2980d-b4e8-49b2-8f45-b8e1be8cea1c" width="400">
</p>

### Confidence measure and diversity loss
The $\mathcal{T}$-similarity and the corresponding diversity loss are implemented as nn.Module which makes them easy to use, e.g., to train the architecture below.

### Learning with the $\mathcal{T}$-similarity
To combine prediction and confidence estimation, e.g., for self-training, we prpropose the lightweight architecture shown below. To train it, we provide a sklearn-ish base_estimator with fit, predict, predict_proba methods and the novel predict_t_similarity method.

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

## Coming Soon
The code is still in development and we will add the following points very soon:
- Clean requirements.txt
- Visualization of ECE for softmax and $\mathcal{T}$-similarity (Figure 5)
- Self-training algorithms

## Modules
SAMformer consists of several key modules:
- `notebooks/`: Contains the notebooks to reproduce the figures from the paper;
- `data/`: Contains the datasets used in our experiments;
- `src/datasets`: Contains the functions to load datasets and perform the labeling procedure;
- `src/models/tsim`: Contains the implementation of the $\mathcal{T}$-similarity and the diversity loss
- `src/models/diverse_ensemble`: Contains the implementation of the global architectures shown above
- `src/models/architectures`: Contains the implementation of neural networks and ensemble classifier;
- `src/models/utils`: Contains custom Datasets and DataLoader

## Installation
To get started with $\mathcal{T}$-similarity, clone this repository and install the required packages (requirements.txt coming soon).

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

## Cite us
If you use our code in your research,  please cite:

```
@InProceedings{pmlr-v238-odonnat24a,
  title = 	 { Leveraging Ensemble Diversity for Robust Self-Training in the Presence of Sample Selection Bias },
  author =       {Odonnat, Ambroise and Feofanov, Vasilii and Redko, Ievgen},
  booktitle = 	 {Proceedings of The 27th International Conference on Artificial Intelligence and Statistics},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v238/odonnat24a/odonnat24a.pdf},
  url = 	 {https://proceedings.mlr.press/v238/odonnat24a.html},
}
```
