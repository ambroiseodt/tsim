# $\mathcal{T}$-similarity
**This is the official implementation of [Leveraging Ensemble Diversity for Robust Self-Training in the Presence of Sample Selection Bias](https://arxiv.org/pdf/2310.14814), AISTATS 2024.**

## Overview
We provide the implementation of the **$\mathcal{T}$-similarity**, a drop-in replacement of the softmax for confidence estimation under distribution shifts. This novel confidence measure is build upon a diverse ensemble of linear classifiers and corrects the softmax overconfidence while being calibrated. It can be used for confidence estimation and SSL methods using neural networks as backbones, e.g., self-training.

<p align="center">
  <img src="https://github.com/ambroiseodt/tsim/assets/64415312/eeef3f8c-308e-4dd6-af3c-58b943634963" width="600">
</p>

## What is included?
We provide the following implementions.

### Labeling procedure
Sample selection bias (SSB) occurs when data labeling is subject to constraints resulting in a distribution mismatch between labeled and unlabeled data.
We illustrate below the two types of labeling considered in our [paper](https://arxiv.org/pdf/2310.14814):
- **IID**: The usual uniform labeling that verifies the i.i.d. assumption;
- **SSB**: Distribution shift between labeled and unlabeled data.

<p align="center">
  <img src="https://github.com/ambroiseodt/tsim/assets/64415312/986153eb-0a1e-4472-9d86-98f7892b7f2c" width="600">
</p>

### Learning with the $\mathcal{T}$-similarity
We provide the PyTorch implementation of the $\mathcal{T}$-similarity and the corresponding diversity loss. To combine prediction and confidence estimation, e.g., for self-training, we introduce the lightweight architecture shown below. In terms of implementation, it has the form of an sklearn base_estimator with fit, predict, and predict_proba methods and we add a predict_t_similarity method.

<p align="center">
  <img src="https://github.com/ambroiseodt/tsim/assets/64415312/797cdfff-0621-420f-bc65-100a50f140cb" width="350">
</p>

#### *Key features*
- Backpropagation of the diversity loss only influences the ensemble, not the projection layers;
- In practice, we use $M=5$ heads resulting in lightweight and fast training;
- Compatible to any SSL methods using neural networks as backbones.

## Examples
We provide demos in `notebooks/` to take in hand the implementation and reproduce the figures of the [paper](https://arxiv.org/pdf/2310.14814):
- `plot_intro_figure.ipynb`: Overview of the method (Figure 1)
- `plot_sample_selection_bias.ipynb`: Visualization of the sample selection bias (Figure 3)
- `plot_calibration.ipynb`: $\mathcal{T}$-similarity corrects overconfidence of the softmax (Figure 6)

The code below (in `plot_calibration.ipynb`) gives an example of how to train the model we introduced with $5$ ensemble heads:
```python
import sys
sys.path.append("..")
from src.datasets.read_dataset import RealDataSet
from src.models.diverse_ensemble import DiverseEnsembleMLP

def get_base_classier(
    dataset_name,
    seed,
    nb_lab_samples_per_class,
    selection_bias,
    num_epochs,
    gamma,
):
    # Fixed params
    test_size = 0.25
    n_iters = 100
    n_classifiers = 5

    # Data split
    dataset = RealDataSet(dataset_name=dataset_name, seed=seed)

    # Percentage of labeled data
    num_classes = len(list(set(dataset.y)))
    ratio = num_classes / ((1 - test_size) * len(dataset.y))
    lab_size = nb_lab_samples_per_class * ratio

    # Split
    x_l, x_u, y_l, y_u, x_test, y_test, n_classes = dataset.get_split(
        test_size=test_size, lab_size=lab_size, selection_bias=selection_bias
    )

    # Define base classifier
    base_classifier = DiverseEnsembleMLP(
        num_epochs=num_epochs,
        gamma=gamma,
        n_iters=n_iters,
        n_classifiers=n_classifiers,
        device="cpu",
        verbose=False,
        random_state=seed,
    )

    # Train
    base_classifier.fit(x_l, y_l, x_u)

    return base_classifier, x_u, y_u
```


## Coming soon
The code is still in development and we will add the following components very soon:
- Visualization of ECE for softmax and $\mathcal{T}$-similarity (Figure 5)
- Self-training algorithms

## Modules
This package consists of several key modules:
- `notebooks/`: Contains the notebooks to reproduce the figures from the paper;
- `data/`: Contains the datasets used in our experiments;
- `src/datasets`: Contains the functions to load datasets and perform the labeling procedure;
- `src/models/`: Contains all the functions to train diverse ensemble with the $\mathcal{T}$-similarity

## Installation
> [!WARNING]
> A clean requirements.txt file will be added soon.

To get started with $\mathcal{T}$-similarity, clone this repository and install the required packages.

```bash
git clone https://github.com/ambroiseodt/tsim.git
cd tsim
pip install -r requirements.txt
```
Please, make sure you have Python 3.8 or a newer version installed.

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
