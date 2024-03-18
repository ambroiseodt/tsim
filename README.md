# T-similarity
Official implementation of [Leveraging Ensemble Diversity for Robust Self-Training in the Presence of Sample Selection Bias](https://arxiv.org/pdf/2310.14814), accepted at AISTATS 2024, Valencia, Spain.

## Code in development
The implementation of the $\mathcal{T}$-similarity and the code to reproduce the experiments of the paper will be available soon. 

## Note
For the MSLA baseline, there is one difference between this implementation and our paper: the supervised loss $\ell_\mathrm{sup}$ of Eq.(1) is obtained by a single average over the labeled training set instead of averaging separately over the originally labeled data and the pseudo-labeled data. We find that it improves the results, accentuating the superiority of the $\mathcal{T}$-similarity over the $\texttt{softmax}$. For the other baselines, the implementation has not changed (single average over the labeled training set).

