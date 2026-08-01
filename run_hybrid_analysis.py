"""
Runs the full Hybrid CNN-BiLSTM-Attention analysis (CV, hold-out,
permutation importance, MC dropout, Kernel SHAP, bootstrap CI) and
writes results to results/hybrid_*.

This is an independent NumPy reimplementation of the architecture and
training protocol described in the paper (Sections 2.7.1, 2.9, 2.14.1,
2.15-2.19); the original from-scratch code was not included in the
project files this was built from. Exact headline numbers will differ
from the manuscript's reported figures — see README_hybrid.md.
"""
import json
import time
import numpy as np
import pandas as pd

from hybrid_model.train import (
    FEATURES_6, TARGET, cross_validate, holdout_split_train,
    permutation_importance_hybrid, mc_dropout_uncertainty, kernel_shap,
    bootstrap_ci,
)

RESULTS_DIR = "results"


def main():
    df = pd.read_csv("data/data_full.csv")
    X = df[FEATURES_6].values
    y = df[TARGET].values
    Xs = (X - X.mean(0)) / X.std(0)

    print("=" * 70)
    print("STAGE 1/5: 5-fold CV x 10 repeats (50 folds)")
    print("=" * 70)
    t0 = time.time()
    cv_df = cross_validate(Xs, y, n_splits=5, n_repeats=10, seed=42,
                            max_epochs=600, patience=40)
    cv_df.to_csv(f"{RESULTS_DIR}/hybrid_cv_raw.csv", index=False)
    cv_summary = {
        "r2_mean": float(cv_df.r2.mean()), "r2_std": float(cv_df.r2.std()),
        "rmse_mean": float(cv_df.rmse.mean()), "rmse_std": float(cv_df.rmse.std()),
        "mae_mean": float(cv_df.mae.mean()), "mae_std": float(cv_df.mae.std()),
    }
    with open(f"{RESULTS_DIR}/hybrid_cv_summary.json", "w") as f:
        json.dump(cv_summary, f, indent=2)
    print(f"CV done in {time.time() - t0:.0f}s. Summary: {cv_summary}")

    print("=" * 70)
    print("STAGE 2/5: 70/15/15 hold-out split, permutation importance")
    print("=" * 70)
    t0 = time.time()
    model, (X_tr, y_tr, X_val, y_val, X_test, y_test), n_ep = holdout_split_train(
        Xs, y, seed=7, max_epochs=600, patience=60)
    pred_test = model.predict(X_test, training=False)

    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    holdout_metrics = {
        "r2": float(r2_score(y_test, pred_test)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred_test))),
        "mae": float(mean_absolute_error(y_test, pred_test)),
        "mape": float(np.mean(np.abs((y_test - pred_test) / y_test)) * 100),
        "n_test": len(y_test), "epochs_trained": n_ep,
        "n_params": model.n_params(),
    }
    with open(f"{RESULTS_DIR}/hybrid_holdout_metrics.json", "w") as f:
        json.dump(holdout_metrics, f, indent=2)
    pd.DataFrame({"observed": y_test, "predicted": np.round(pred_test, 4)}).to_csv(
        f"{RESULTS_DIR}/hybrid_holdout_compare.csv", index=False)
    print(f"Hold-out metrics: {holdout_metrics}  ({time.time()-t0:.0f}s)")

    importances, base_r2 = permutation_importance_hybrid(
        model, X_test, y_test, FEATURES_6, n_repeats=30, seed=42)
    with open(f"{RESULTS_DIR}/hybrid_importance.json", "w") as f:
        json.dump(importances, f, indent=2)
    print(f"Permutation importance: {importances}")

    print("=" * 70)
    print("STAGE 3/5: Monte Carlo dropout uncertainty (M=200)")
    print("=" * 70)
    t0 = time.time()
    mc = mc_dropout_uncertainty(model, X_test, y_test, M=200, seed=42)
    with open(f"{RESULTS_DIR}/hybrid_mc_dropout.json", "w") as f:
        json.dump(mc, f, indent=2)
    print(f"MC dropout empirical coverage: {mc['empirical_coverage']:.3f}  ({time.time()-t0:.0f}s)")

    print("=" * 70)
    print("STAGE 4/5: Kernel SHAP (150 coalitions, background=20)")
    print("=" * 70)
    t0 = time.time()
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(len(X_tr), size=min(20, len(X_tr)), replace=False)
    X_background = X_tr[bg_idx]
    shap_values = kernel_shap(model, X_test, X_background, n_coalitions=150, seed=42)
    np.save(f"{RESULTS_DIR}/hybrid_shap_values.npy", shap_values)
    shap_mean_abs = {name: float(np.mean(np.abs(shap_values[:, j])))
                      for j, name in enumerate(FEATURES_6)}
    with open(f"{RESULTS_DIR}/hybrid_shap_summary.json", "w") as f:
        json.dump(shap_mean_abs, f, indent=2)
    print(f"Mean |SHAP|: {shap_mean_abs}  ({time.time()-t0:.0f}s)")

    print("=" * 70)
    print("STAGE 5/5: Bootstrap 95% CI (B=5000) on hold-out metrics")
    print("=" * 70)
    t0 = time.time()
    ci = bootstrap_ci(y_test, pred_test, B=5000, seed=42)
    with open(f"{RESULTS_DIR}/hybrid_bootstrap_ci.json", "w") as f:
        json.dump(ci, f, indent=2)
    print(f"Bootstrap CI: {ci}  ({time.time()-t0:.0f}s)")

    print("\nAll hybrid-model analysis artifacts written to results/hybrid_*")


if __name__ == "__main__":
    main()
