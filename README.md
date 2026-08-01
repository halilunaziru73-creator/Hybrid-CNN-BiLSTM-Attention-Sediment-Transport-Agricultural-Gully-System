# A Hybrid CNN-BiLSTM-Attention Deep Learning Framework for Sediment Transport in an Agricultural Gully System

**Author:** Naziru Halilu

This repository accompanies the manuscript (`manuscript/Halilu_Sediment_Transport.docx`),
comparing benchmark MLP/linear-regression models against a Hybrid
CNN-BiLSTM-Attention model for predicting sediment transport rate in an
agricultural gully system.

## Figures

### Data & Feature Overview

![Distribution of the sediment transport target variable](figures/fig_hist_target.png)
**Figure 3A** — Distribution of the sediment transport target variable

![Feature correlation heatmap](figures/fig_corr_heatmap.png)
**Figure 3B** — Feature correlation heatmap

![Feature vs. target scatter grid](figures/fig_feature_scatter_grid.png)
**Figure 3C** — Feature vs. target scatter grid, with Pearson r

![Performance by field scenario](figures/fig_boxplot_scenario.png)
**Figure 3D** — Distribution of sediment transport rate by field scenario

### Benchmark Model (MLP / Linear) Cross-Validation

![Cross-validated R² by model (bar)](figures/fig_cv_r2_bar.png)
**Figure 4A** — Mean cross-validated R² ± SD across models

![Cross-validated R² by model (boxplot, 50 folds)](figures/fig_cv_r2_boxplot.png)
**Figure 4B** — Distribution of fold-wise R² (50 folds)

![Cross-validated RMSE / MAE by model](figures/fig_cv_rmse_mae_bar.png)
**Figure 4C** — Mean RMSE and MAE across folds

### Hold-out Performance

![Hold-out predicted vs. observed (6-variable model)](figures/fig_holdout_pred_obs_6feat.png)
**Figure 5A** — Predicted vs. observed, six-variable benchmark MLP (dashed = 1:1 line)

![Hold-out residuals](figures/fig_holdout_residual.png)
**Figure 5B** — Residuals vs. observed

![Hold-out error distribution](figures/fig_holdout_error_hist.png)
**Figure 5C** — Distribution of absolute prediction errors

![Hold-out predictions across sample index](figures/fig_holdout_index_line.png)
**Figure 5D** — Observed and predicted values, ranked from highest to lowest observed

### Feature Importance

![Permutation feature importance](figures/fig_importance_bar.png)
**Figure 6A** — Permutation importance, six-variable model

![Feature importance: 4-variable vs. 6-variable model](figures/fig_importance_4v6.png)
**Figure 6B** — Shared hydraulic variables in the four- vs. six-variable models

![Feature importance (Pareto)](figures/fig_importance_pareto.png)
**Figure 6C** — Pareto chart of six-variable predictor importance

### Model Comparison Summary

![Grouped performance comparison, all models](figures/fig_grouped_bar_all_models.png)
**Figure 7A** — Cross-validated R², RMSE and MAE across all four models

![Model metrics radar comparison](figures/fig_metrics_radar.png)
**Figure 7B** — Normalised cross-validated performance profile

![Predicted vs. observed overlay, all models](figures/fig_comparison_scatter_overlay.png)
**Figure 7C** — Hold-out predictions of the six-variable deep-learning and linear
models overlaid against observed values

### Hybrid CNN-BiLSTM-Attention Model

![Hybrid CNN-BiLSTM-Attention network architecture](figures/fig_network_architecture.png)
**Figure 2** — Feedforward MLP benchmark architecture (6-8-4-1 configuration)

![Hybrid model: hold-out and CV performance](figures/fig_hybrid_holdout_and_cv.png)
**Figure 10** — Hybrid model performance: predicted vs. observed, residuals, and
cross-validated R² comparison across the linear, MLP, and hybrid models

![Hybrid model: importance, SHAP, and Monte Carlo dropout uncertainty](figures/fig_hybrid_importance_shap_mcdropout.png)
**Figure 11** — Hybrid model interpretability and uncertainty: permutation
importance, Kernel SHAP summary, and Monte Carlo dropout prediction intervals

---

## Contents

- `data/`
  - `table41.csv` — channel geometric survey (breadth, depth1, depth2, average
    depth, slope) at 20 stations.
  - `data.csv` — the 100-observation hydraulic dataset (depth, slope, soil shear,
    sediment transport rate, velocity) across 5 field scenarios.
  - `data_full.csv` — merged dataset (100 rows) combining the hydraulic dataset
    with channel breadth and average channel depth, matched spot-by-spot via
    slope alignment. This is the dataset used to train all models.

- `full_analysis.py` — main analysis script for the benchmark models: trains and
  cross-validates the 6-variable and 4-variable MLP and linear-regression models,
  computes hold-out performance, and computes permutation importance.

- `hybrid_model/` — the NumPy implementation of the Hybrid CNN-BiLSTM-Attention
  model (Section 2.7.1 of the manuscript), including a gradient-check module for
  verifying backward-pass correctness. See `hybrid_model/README_hybrid.md` for
  the full architectural documentation.

- `run_hybrid_analysis.py` — runs the full hybrid-model cross-validation,
  hold-out, permutation-importance, Monte Carlo dropout, and Kernel SHAP
  pipeline. Writes `results/hybrid_*`.

- `make_charts.py` — generates the benchmark-model (MLP/linear) figures.
- `make_hybrid_charts.py` — generates the hybrid-model figures (hold-out/CV
  comparison; importance/SHAP/Monte Carlo dropout).

- `figures/` — all figures referenced in the manuscript.

- `results/` — numeric outputs from both the benchmark and hybrid model
  pipelines: cross-validation results, hold-out metrics, permutation importance,
  Kernel SHAP values, Monte Carlo dropout distributions, and bootstrap
  confidence intervals.

## Reproducing the results

```bash
pip install scikit-learn pandas numpy scipy matplotlib --break-system-packages

# Benchmark MLP / linear regression models
python3 full_analysis.py
python3 make_charts.py

# Hybrid CNN-BiLSTM-Attention model (~15 min)
python3 -m hybrid_model.gradcheck      # verify backward-pass correctness first
python3 run_hybrid_analysis.py
python3 make_hybrid_charts.py
```

## Model summary

- **Benchmark architecture:** feedforward multilayer perceptron, 6 (or 4) input
  neurons → 8 neurons (ReLU) → 4 neurons (ReLU) → 1 linear output neuron.
  L-BFGS optimiser, L2 regularisation (α = 0.1).
- **Primary architecture (hybrid):** Conv1D(1→8) → Conv1D(8→16) →
  BiLSTM(8/direction) → attention pooling → Dense(16, ReLU) + Dropout(0.2) →
  linear output. Weighted Huber loss, L2 (α = 1e-3), Adam optimiser. See
  `hybrid_model/README_hybrid.md` for the full specification.
- **Evaluation:** 5-fold cross-validation repeated 10 times (50 folds total),
  an independent hold-out split, and permutation importance for both model
  families; the hybrid model additionally includes Monte Carlo dropout
  uncertainty and Kernel SHAP.
- **Headline results:** benchmark MLP cross-validated R² = 0.62 ± 0.37,
  hold-out R² = 0.91 (6-variable); hybrid model cross-validated R² = 0.68 ± 0.36,
  hold-out R² = 0.90. See the manuscript for full details, governing equations,
  and discussion.

## License

Released under the [MIT License](./LICENSE).
