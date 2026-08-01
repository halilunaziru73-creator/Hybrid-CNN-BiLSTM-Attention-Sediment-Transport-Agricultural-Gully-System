import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json

np.random.seed(42)

df = pd.read_csv("data/data_full.csv")
print("Dataset shape:", df.shape)

FEATURES_4 = ["depth_ft", "slope", "soil_shear_lbft2", "velocity_ms"]
FEATURES_6 = ["depth_ft", "slope", "soil_shear_lbft2", "velocity_ms", "breadth_m", "channel_avg_depth_m"]
TARGET = "sed_transport_rate_kgsm"

y = df[TARGET].values

def evaluate(feature_list, model_builder, n_splits=5, n_repeats=10, seed=42):
    X = df[feature_list].values
    kf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    r2s, rmses, maes = [], [], []
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        scaler = StandardScaler().fit(X_train)
        X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)
        model = model_builder()
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)
        r2s.append(r2_score(y_test, pred))
        rmses.append(np.sqrt(mean_squared_error(y_test, pred)))
        maes.append(mean_absolute_error(y_test, pred))
    return {
        "r2_mean": float(np.mean(r2s)), "r2_std": float(np.std(r2s)),
        "rmse_mean": float(np.mean(rmses)), "rmse_std": float(np.std(rmses)),
        "mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes)),
    }

def mlp_builder():
    return MLPRegressor(hidden_layer_sizes=(8, 4), activation='relu', solver='lbfgs',
                         alpha=0.1, max_iter=5000, random_state=42)

def lin_builder():
    return LinearRegression()

results = {
    "mlp_4feat":  evaluate(FEATURES_4, mlp_builder),
    "mlp_6feat":  evaluate(FEATURES_6, mlp_builder),
    "linear_4feat": evaluate(FEATURES_4, lin_builder),
    "linear_6feat": evaluate(FEATURES_6, lin_builder),
}
print(json.dumps(results, indent=2))

with open("results/cv_results_full.json", "w") as f:
    json.dump(results, f, indent=2)

# ---- Final 6-feature deep learning model: 80/20 hold-out for reporting ----
X6 = df[FEATURES_6].values
X_train, X_test, y_train, y_test = train_test_split(X6, y, test_size=0.2, random_state=7)
scaler = StandardScaler().fit(X_train)
X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

final_mlp = mlp_builder()
final_mlp.fit(X_train_s, y_train)
pred_test = final_mlp.predict(X_test_s)

holdout_metrics = {
    "r2": float(r2_score(y_test, pred_test)),
    "rmse": float(np.sqrt(mean_squared_error(y_test, pred_test))),
    "mae": float(mean_absolute_error(y_test, pred_test)),
    "n_test": int(len(y_test)),
}
print("\n6-feature hold-out metrics:", holdout_metrics)

out = pd.DataFrame({
    "observed_kg_s_m": y_test,
    "predicted_kg_s_m": np.round(pred_test, 3)
})
out.to_csv("results/holdout_predictions_6feat.csv", index=False)
print(out)

with open("results/holdout_metrics_6feat.json", "w") as f:
    json.dump(holdout_metrics, f, indent=2)

# Feature importance proxy: permutation importance on hold-out set
from sklearn.inspection import permutation_importance
perm = permutation_importance(final_mlp, X_test_s, y_test, n_repeats=30, random_state=42, scoring="r2")
importances = {feat: {"mean": float(m), "std": float(s)} for feat, m, s in
               zip(FEATURES_6, perm.importances_mean, perm.importances_std)}
print("\nPermutation importances (R2 drop):", json.dumps(importances, indent=2))
with open("results/feature_importance.json", "w") as f:
    json.dump(importances, f, indent=2)
