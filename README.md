# A Hybrid CNN-BiLSTM-Attention Deep Learning Framework for Sediment Transport in an Agricultural Gully System

This repository accompanies the manuscript (see `manuscript/Halilu_Sediment_Transport.docx`), 
comparing benchmark MLP/linear-regression models against a from-scratch NumPy reimplementation 
of a Hybrid CNN-BiLSTM-Attention model for predicting sediment transport rate in an agricultural gully system.

## Figures

### Data & Feature Overview

![Distribution of the sediment transport target variable](figures/fig_hist_target.png)
**fig_hist_target.png** — Distribution of the sediment transport target variable

![Feature correlation heatmap](figures/fig_corr_heatmap.png)
**fig_corr_heatmap.png** — Feature correlation heatmap

![Feature vs. target scatter grid](figures/fig_feature_scatter_grid.png)
**fig_feature_scatter_grid.png** — Feature vs. target scatter grid

### Benchmark Model (MLP / Linear) Cross-Validation

![Cross-validated R² by model (bar)](figures/fig_cv_r2_bar.png)
**fig_cv_r2_bar.png** — Cross-validated R² by model (bar)

![Cross-validated R² by model (boxplot, 50 folds)](figures/fig_cv_r2_boxplot.png)
**fig_cv_r2_boxplot.png** — Cross-validated R² by model (boxplot, 50 folds)

![Cross-validated RMSE / MAE by model](figures/fig_cv_rmse_mae_bar.png)
**fig_cv_rmse_mae_bar.png** — Cross-validated RMSE / MAE by model

![Fold-wise cross-validation performance](figures/fig_cv_foldwise_line.png)
**fig_cv_foldwise_line.png** — Fold-wise cross-validation performance

![Grouped performance comparison, all models](figures/fig_grouped_bar_all_models.png)
**fig_grouped_bar_all_models.png** — Grouped performance comparison, all models

### Feature Importance

![Permutation feature importance](figures/fig_importance_bar.png)
**fig_importance_bar.png** — Permutation feature importance

![Feature importance (Pareto)](figures/fig_importance_pareto.png)
**fig_importance_pareto.png** — Feature importance (Pareto)

![Feature importance: 4-variable vs. 6-variable model](figures/fig_importance_4v6.png)
**fig_importance_4v6.png** — Feature importance: 4-variable vs. 6-variable model

### Hold-out Performance

![Hold-out predicted vs. observed (6-variable model)](figures/fig_holdout_pred_obs_6feat.png)
**fig_holdout_pred_obs_6feat.png** — Hold-out predicted vs. observed (6-variable model)

![Hold-out predictions across sample index](figures/fig_holdout_index_line.png)
**fig_holdout_index_line.png** — Hold-out predictions across sample index

![Hold-out residuals](figures/fig_holdout_residual.png)
**fig_holdout_residual.png** — Hold-out residuals

![Hold-out error distribution](figures/fig_holdout_error_hist.png)
**fig_holdout_error_hist.png** — Hold-out error distribution

![Predicted vs. observed overlay, all models](figures/fig_comparison_scatter_overlay.png)
**fig_comparison_scatter_overlay.png** — Predicted vs. observed overlay, all models

![Performance by field scenario](figures/fig_boxplot_scenario.png)
**fig_boxplot_scenario.png** — Performance by field scenario

![Model metrics radar comparison](figures/fig_metrics_radar.png)
**fig_metrics_radar.png** — Model metrics radar comparison

### Hybrid CNN-BiLSTM-Attention Model

![Hybrid CNN-BiLSTM-Attention network architecture](figures/fig_network_architecture.png)
**fig_network_architecture.png** — Hybrid CNN-BiLSTM-Attention network architecture

![Hybrid model: hold-out and CV performance](figures/fig_hybrid_holdout_and_cv.png)
**fig_hybrid_holdout_and_cv.png** — Hybrid model: hold-out and CV performance

![Hybrid model: importance, SHAP, and Monte Carlo dropout uncertainty](figures/fig_hybrid_importance_shap_mcdropout.png)
**fig_hybrid_importance_shap_mcdropout.png** — Hybrid model: importance, SHAP, and Monte Carlo dropout uncertainty

---


# Deep Learning Sediment Transport Model — Code & Data Package

## Contents

- `data/`
  - `table41.csv` — channel geometric survey (breadth, depth1, depth2, average depth, slope) at 20 spots, digitised from Table 4.1.
  - `data.csv` — the 100-observation hydraulic dataset (depth, slope, soil shear, sediment transport rate, velocity) digitised from Tables 4.2–4.6, across 5 field scenarios.
  - `data_full.csv` — merged dataset (100 rows) combining the hydraulic dataset with channel breadth and average channel depth from the geometric survey, matched spot-by-spot via slope alignment. This is the dataset used to train all models.

- `full_analysis.py` — main analysis script for the benchmark models. Trains and cross-validates the 6-variable and 4-variable MLP and linear-regression models, computes hold-out performance, and computes permutation importance. Writes all result files used in `results/`.

- `train_model.py`, `train_model_full.py` — earlier/intermediate training scripts (4-variable-only and first 6-variable version); kept for transparency of the model development process. `full_analysis.py` is the final, complete version.

- `hybrid_model/` — **independent NumPy reimplementation of the paper's primary Hybrid CNN-BiLSTM-Attention model** (Section 2.7.1). The original from-scratch implementation was not available in this project's files; see `hybrid_model/README_hybrid.md` for exactly what was reconstructed from the manuscript's equations, what was gradient-checked, and where this implementation's numbers differ from — and now supersede — the original manuscript text.

- `run_hybrid_analysis.py` — runs the full hybrid-model CV / hold-out / permutation-importance / Monte Carlo dropout / Kernel SHAP / bootstrap-CI pipeline. Writes `results/hybrid_*`.

- `make_charts.py` — generates the 19 benchmark-model (MLP/linear) figures used in the paper. Reads from `data/data_full.csv` and `results/`.

- `make_hybrid_charts.py` — generates the 2 hybrid-model figures (hold-out/CV comparison; importance/SHAP/MC-dropout). Reads from `results/hybrid_*`.

- `figures/` — all 21 PNG charts embedded in the paper (Figures 1–11; some figures have multiple panels generated by one script).

- `results/` — numeric outputs:
  - `cv_raw_*.csv`, `cv_results_all.json`, `holdout_compare.csv`, `holdout_all_metrics.json`, `importance_*.json` — benchmark MLP/linear results (from `full_analysis.py`).
  - `hybrid_*` — hybrid-model results (from `run_hybrid_analysis.py`): CV, hold-out, permutation importance, Kernel SHAP, Monte Carlo dropout, bootstrap CIs.

- `build_paper_v3.js` — Node.js script (uses the `docx` npm package) that assembles the Word document from text, tables, equations and figures. Run with `node build_paper_v3.js` after `npm install docx`.

## Reproducing the results

```bash
pip install scikit-learn pandas numpy scipy matplotlib --break-system-packages

# Benchmark MLP / linear regression models
python3 full_analysis.py
python3 make_charts.py

# Hybrid CNN-BiLSTM-Attention model (independent NumPy implementation; ~15 min)
python3 -m hybrid_model.gradcheck      # verify backward-pass correctness first
python3 run_hybrid_analysis.py
python3 make_hybrid_charts.py

# Rebuild the Word document (optional)
npm install docx
node build_paper_v3.js
```

## Model summary

- **Benchmark architecture:** feedforward multilayer perceptron, 6 (or 4) input neurons → 8 neurons (ReLU) → 4 neurons (ReLU) → 1 linear output neuron. L-BFGS optimiser, L2 regularisation (α = 0.1).
- **Primary architecture (hybrid):** Conv1D(1→8)→Conv1D(8→16)→BiLSTM(8/direction)→attention pooling→Dense(16, ReLU)+Dropout(0.2)→linear output. Weighted Huber loss, L2 (α=1e-3), Adam. See `hybrid_model/README_hybrid.md`.
- **Evaluation:** 5-fold cross-validation repeated 10 times (50 folds total) + an independent hold-out split + permutation importance, for both model families; the hybrid model additionally gets Monte Carlo dropout uncertainty and Kernel SHAP.
- **Headline results (paper text now matches these):** benchmark MLP cross-validated R² = 0.62 ± 0.37, hold-out R² = 0.91 (6-variable); hybrid model cross-validated R² = 0.68 ± 0.36, hold-out R² = 0.90. See the paper for full details, governing equations, and discussion.

