# Synthetic-Control-Paper-Model

This repo shows the source code for experiments in Synthetic Control Method with Many Outcomes, currently pending revies in JMLR (Journal of Machine Learning Research)

The methodology in this paper introduces a novel concept of applying Bayesian models in tackling synthetic control, a popular method for analyzing panel data to estimate causal effects of major policy interventions.

We show that this method can identify causal effects in non-linear systems by analyzing a recent event of opening footbal stadiums during the COVID-19 Pandemic [Garcia Bulle et all., 2022]


For NFL stadium counties that opened their stadiums during COVID, we construct a synthetic 'control' dataset that serves as the likely trajectory of cases had they not opened, using the following bayesian factor models: Probabilistic PCA Factorization, and Gamma-Poisson Factorization. We compare these two models with traditional robust synthetic control, and show that they perform better on the placebo-check stadiums, and are more robust to continuity issues in the panel data. 

Results: 

![plots_now_allowed](https://github.com/user-attachments/assets/03239cd9-ad94-4a4b-a2b4-1cec7ccb893b)

![plots_allowed](https://github.com/user-attachments/assets/8223e504-a21b-4109-88e4-e52863f0bea2)


[TTT.pdf](https://github.com/user-attachments/files/18544183/TTT.pdf)
