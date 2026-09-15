# Mathematical consolidation — reproducible research pilot

This additive pilot expands the generic provider. It does not change any consumer's canonical inference population, acceptance criteria, or authority.

Run `python -m pip install numpy scipy plotly`, then `python research/consolidation_lab.py` from the repository root. The receipt records exact installed versions. Output is deterministic within that environment. `dashboard.html` embeds Plotly for offline use.

The generated population has 180 independent synthetic observations and 9 features with three latent blocks. It is not a recovered historical dataset. The original QPS bridge numbering is preserved below.

| Original bridge item | Implementation coverage in this pilot | Remaining consumer work |
|---|---|---|
| 1 Standardization | Finite/constant guards and sample standard deviation | Govern feature semantics and missingness |
| 2 Covariance/correlation | Matrix and heatmap | Compare source-bound feature blocks |
| 3 PCA | SVD/eigh reference, PA95, bootstrap eigenvalue intervals | Govern retention; align bootstrap eigenspaces |
| 4 Loadings | Correlation loadings and separate 3D score plot | Calibrate semantic labels |
| 5 Mahalanobis | Pseudoinverse diagnostic | Reference population and anomaly threshold |
| 6 Normalized proximity | Spectral ratios available in receipt | Govern limits, units and polarity |
| 7 Convergence | Distance, contraction, derivatives and integral | Bind real elapsed time and frozen state |
| 8 Quadratic penalties | Mahalanobis is one PSD quadratic form | Explicit general consumer weight contract |
| 9 Feature blocks | Synthetic three-block construction | Bind real theme/topic/item dataset |
| 10 Presentation | Interactive HTML research surface | Production styling and parity integration |
| 11 Federation boundary | Synthetic label, zero authority and credit | External exact-source consumer proof |

Additional execution includes one-way ANOVA on a predefined synthetic response, a reliability Monte Carlo scenario, explicit synthetic paired-outcome Bradley–Terry fitting using the existing kernel, and a four-node path Laplacian with exact spectrum reference.

The weight iteration is a damped move toward fixed retained-variance contributions. It is not supervised learning, a proof of optimal weights, or an authorization to change engineering weights. PCA signs do not supply utility direction. The BT fixture deliberately contains both win directions for each pair; the existing generic kernel's undirected connectivity guard alone does not establish existence of a finite unregularized MLE for arbitrary inputs.

PC1–PC3 is principal-component coordinate space, not a guarantee of a normal distribution. The spectral surface is component × bootstrap replicate × eigenvalue, not a physical surface and not a temporal trajectory. PA95 is a null percentile, not a parameter confidence interval. Bootstrap ordered eigenvalue intervals are descriptive and can be unreliable at ties or poorly separated eigenvalues.

Five mathematical legs: (1) matrix/spectral PCA, (2) paired comparison/ranking, (3) covariance and inferential ANOVA, (4) Monte Carlo/uncertainty, (5) graph/spectral dependencies. ML and temporal orchestration cross these legs. DNN/CNN/graph neural network training, real production data, model fusion, held-out validation, and full federation remain future scoped work.

Post-2000 reading anchors:

- Halko, Martinsson & Tropp (2011), randomized low-rank matrix decompositions: https://tropp.caltech.edu/papers/HMT11-Finding-Structure.pdf
- Candès et al. (2011), robust PCA, low-rank plus sparse recovery: https://people.eecs.berkeley.edu/~yima/psfile/JACM11.pdf
- Yu, Wang & Samworth (2015; preprint 2014), Davis–Kahan eigenspace perturbation: https://arxiv.org/abs/1405.0680
- Gavish & Donoho (2014), singular-value hard thresholds under specified noise/aspect assumptions: https://arxiv.org/abs/1305.5870
- von Luxburg (2007), graph Laplacians and spectral clustering: https://www.cs.columbia.edu/~jebara/6772/papers/Luxburg07_tutorial.pdf
- Tu et al. (2014; preprint 2013), dynamic mode decomposition for paired snapshots: https://arxiv.org/abs/1312.0041
- Kipf & Welling (2017; preprint 2016), graph convolutional networks: https://arxiv.org/abs/1609.02907
- Gorishniy et al. (2021), tabular neural baselines and FT-Transformer: https://proceedings.nips.cc/paper_files/paper/2021/hash/9d86d83f925f2149e9edb0ac3b49229c-Abstract.html

The bibliography is an initial verified reading set, not an exhaustive review. No paper illustrations are redistributed.
