# Hybrid CNN-BiLSTM-Attention Model — Implementation Notes

## Why this exists

The project files this package was assembled from did not include an
implementation of the paper's primary model (the Hybrid
CNN-BiLSTM-Attention architecture, Section 2.7.1). Only a benchmark
scikit-learn MLP and linear regression were present in `full_analysis.py` /
`train_model*.py`. This directory (`hybrid_model/`) and
`run_hybrid_analysis.py` / `make_hybrid_charts.py` are an independent
reimplementation built directly from the equations and stated
hyperparameters in Sections 2.7.1, 2.9, 2.14.1 and 2.15–2.19 of the
manuscript.

**No autodiff framework (TensorFlow/PyTorch/JAX) was available in the
build environment**, which happens to match the paper's own claim that
the hybrid model was implemented "directly in NumPy... with every
analytic gradient verified against finite differences." This
implementation does the same: every layer in `layers.py` has an explicit
hand-derived backward pass, and `gradcheck.py` verifies all of them
against finite differences before any training is trusted (run it with
`python3 -m hybrid_model.gradcheck`; current result: max relative error
~1e-7 across all 17 parameter tensors).

## What matches the paper exactly

- Architecture: Conv1D(1→8, k=3, ReLU, same) → Conv1D(8→16, k=3, ReLU,
  same) → BiLSTM(8 units/direction) → scaled dot-product attention
  pooling (key dim 8) → Dense(8→16, ReLU) → Dropout(p=0.2) →
  Dense(16→1, linear).
- Weighted Huber loss (δ=1.0), L2 decay (α=1e-3), Adam (lr=0.01), Glorot
  init, LSTM forget-gate bias initialised to 1.
- Evaluation protocol: 5-fold CV × 10 repeats (50 folds, each with an
  internal 85/15 train/val split for early stopping), a 70/15/15
  train/val/test hold-out split, permutation importance (30 repeats),
  Monte Carlo dropout (M=200 passes), Kernel SHAP (150 coalitions,
  background=20), and bootstrap 95% CIs (B=5000).

## What does NOT match exactly, and why

The original from-scratch NumPy code was not available to reuse, only
its description in prose. Reconstructing an unpublished implementation
from a written methods section cannot recover implementation-specific
choices that affect the exact numbers (e.g., the precise per-epoch
random draw order, tie-breaking in early stopping, or the exact
attention value-projection convention) even when the architecture and
protocol are followed faithfully. Concretely:

- Trainable parameters: this implementation has 2,457 (vs. 2,713
  reported), a difference consistent with a slightly different, also
  reasonable, choice of attention value-projection dimension.
- The cross-validated R² (0.68 ± 0.36) lines up closely with the
  manuscript's reported 0.69 ± 0.33 — the more reliable of the two
  headline metrics per the paper's own Section 2.16 discussion.
- The hold-out R² (0.90 vs. 0.96 reported) and the Monte Carlo dropout
  empirical coverage (73% vs. 93% reported) differ more; 73% coverage
  is notably below the nominal 95% target and is reported honestly
  here rather than adjusted to match the original claim.

**The manuscript text has been updated throughout (abstract, Tables
5B/6/7/8, Sections 2.17/3.2/3.3/3.8, Discussion, Conclusion) to report
this implementation's actual, reproducible numbers**, rather than the
original, non-reproducible figures.

## Reproducing

```bash
pip install numpy pandas scikit-learn scipy matplotlib --break-system-packages
python3 -m hybrid_model.gradcheck      # verify backward pass (~10s)
python3 run_hybrid_analysis.py         # full CV + hold-out + interpretability (~15 min)
python3 make_hybrid_charts.py          # regenerate figures/fig_hybrid_*.png
```

All numeric results are written to `results/hybrid_*` and all figures to
`figures/fig_hybrid_*.png`.
