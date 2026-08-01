"""
Training / evaluation protocol for the Hybrid CNN-BiLSTM-Attention model,
following paper Sections 2.9, 2.14.1, 2.15-2.19:

 - 5-fold CV repeated 10 times (50 folds), each fold trained with an
   internal 85/15 train/validation split for early stopping.
 - An independent 70/15/15 train/validation/test hold-out split.
 - Permutation importance (30 repeats) on the hold-out test set.
 - Monte Carlo dropout uncertainty (M=200 passes) on the hold-out test set.
 - Kernel SHAP approximation (150 sampled coalitions, background=20).
 - Bootstrap 95% CIs (B=5000) on hold-out R²/RMSE/MAE.

This is an independent reimplementation from the paper's equations, not
a byte-for-byte reproduction of the original (unavailable) code, so exact
headline numbers will differ from the manuscript; see README_hybrid.md.
"""
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .model import HybridCNNBiLSTMAttention

FEATURES_6 = ["depth_ft", "slope", "soil_shear_lbft2", "velocity_ms",
              "breadth_m", "channel_avg_depth_m"]
TARGET = "sed_transport_rate_kgsm"


def train_with_early_stopping(X_train, y_train, X_val, y_val, seed,
                               max_epochs=600, patience=40, verbose=False):
    model = HybridCNNBiLSTMAttention(seed=seed)
    best_val = np.inf
    best_params = model.get_flat_params()
    epochs_no_improve = 0
    rng = np.random.default_rng(seed + 1000)
    n = len(X_train)
    for epoch in range(max_epochs):
        order = rng.permutation(n)
        for i in order:
            _, grads, _ = model.loss_and_grads(X_train[i], y_train[i], training=True)
            model.adam_step(grads)
        val_pred = model.predict(X_val, training=False)
        val_mse = mean_squared_error(y_val, val_pred)
        if val_mse < best_val - 1e-6:
            best_val = val_mse
            best_params = model.get_flat_params()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
    model.set_flat_params(best_params)
    return model, epoch + 1


def cross_validate(X, y, n_splits=5, n_repeats=10, seed=42, max_epochs=600, patience=40):
    kf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    rows = []
    t_start = time.time()
    for fold_i, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train_full, y_train_full = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.15, random_state=fold_i)
        model, n_ep = train_with_early_stopping(
            X_tr, y_tr, X_val, y_val, seed=fold_i, max_epochs=max_epochs, patience=patience)
        pred = model.predict(X_test, training=False)
        rows.append({
            "fold": fold_i, "repeat": fold_i // n_splits,
            "r2": r2_score(y_test, pred),
            "rmse": np.sqrt(mean_squared_error(y_test, pred)),
            "mae": mean_absolute_error(y_test, pred),
            "epochs": n_ep,
        })
        if (fold_i + 1) % 10 == 0:
            elapsed = time.time() - t_start
            print(f"  fold {fold_i + 1}/{n_splits * n_repeats}  "
                  f"elapsed={elapsed:.0f}s  last_r2={rows[-1]['r2']:.3f}  "
                  f"epochs={n_ep}")
    return pd.DataFrame(rows)


def holdout_split_train(X, y, seed=7, max_epochs=600, patience=60):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=seed)
    model, n_ep = train_with_early_stopping(
        X_train, y_train, X_val, y_val, seed=seed, max_epochs=max_epochs, patience=patience)
    return model, (X_train, y_train, X_val, y_val, X_test, y_test), n_ep


def permutation_importance_hybrid(model, X_test, y_test, feature_names, n_repeats=30, seed=42):
    rng = np.random.default_rng(seed)
    base_pred = model.predict(X_test, training=False)
    base_r2 = r2_score(y_test, base_pred)
    importances = {}
    for j, name in enumerate(feature_names):
        drops = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            rng.shuffle(X_perm[:, j])
            pred = model.predict(X_perm, training=False)
            drops.append(base_r2 - r2_score(y_test, pred))
        importances[name] = {"mean": float(np.mean(drops)), "std": float(np.std(drops))}
    return importances, base_r2


def mc_dropout_uncertainty(model, X_test, y_test, M=200, seed=42):
    rng = np.random.default_rng(seed)
    n_test = len(X_test)
    preds = np.zeros((M, n_test))
    dense_units = model.dense1.b.shape[0]
    for m in range(M):
        mask = (rng.uniform(size=dense_units) > model.dropout.p).astype(float) / (1 - model.dropout.p)
        for i, x in enumerate(X_test):
            preds[m, i], _ = model.forward(x, training=True, dropout_mask=mask)
    mean_pred = preds.mean(axis=0)
    lo = np.percentile(preds, 2.5, axis=0)
    hi = np.percentile(preds, 97.5, axis=0)
    coverage = float(np.mean((y_test >= lo) & (y_test <= hi)))
    return {"mean": mean_pred.tolist(), "lower": lo.tolist(), "upper": hi.tolist(),
            "empirical_coverage": coverage}


def kernel_shap(model, X_test, X_background, n_coalitions=150, seed=42):
    """Weighted-least-squares Kernel SHAP approximation (Eq. 36), background=20
    randomly sampled training observations, local accuracy sum(phi)=f(x)-E[f(X)]."""
    rng = np.random.default_rng(seed)
    n_feat = X_test.shape[1]
    ref_mean_pred = model.predict(X_background, training=False).mean()

    def f_of_coalition(x, mask):
        x_masked = np.where(mask, x, X_background)
        preds = model.predict(x_masked, training=False)
        return preds.mean()

    all_phi = np.zeros((len(X_test), n_feat))
    for idx, x in enumerate(X_test):
        Z, weights, fvals = [], [], []
        for _ in range(n_coalitions):
            size = rng.integers(1, n_feat)  # avoid empty/full for weight stability
            subset = rng.choice(n_feat, size=size, replace=False)
            mask = np.zeros(n_feat, dtype=bool)
            mask[subset] = True
            from math import comb
            s = mask.sum()
            w = (n_feat - 1) / (comb(n_feat, s) * s * (n_feat - s)) if 0 < s < n_feat else 1e6
            Z.append(mask.astype(float))
            weights.append(w)
            fvals.append(f_of_coalition(x, mask))
        Z = np.array(Z)
        weights = np.array(weights)
        fvals = np.array(fvals) - ref_mean_pred
        Wm = np.diag(weights)
        A = Z.T @ Wm @ Z + 1e-6 * np.eye(n_feat)
        b = Z.T @ Wm @ fvals
        phi = np.linalg.solve(A, b)
        all_phi[idx] = phi
    return all_phi


def bootstrap_ci(y_true, y_pred, B=5000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    r2s, rmses, maes = [], [], []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        if np.all(yt == yt[0]):
            continue
        r2s.append(r2_score(yt, yp))
        rmses.append(np.sqrt(mean_squared_error(yt, yp)))
        maes.append(mean_absolute_error(yt, yp))
    def ci(a):
        return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))
    return {
        "r2": {"point": float(r2_score(y_true, y_pred)), "ci95": ci(r2s)},
        "rmse": {"point": float(np.sqrt(mean_squared_error(y_true, y_pred))), "ci95": ci(rmses)},
        "mae": {"point": float(mean_absolute_error(y_true, y_pred)), "ci95": ci(maes)},
    }
