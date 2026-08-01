import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
import json

np.random.seed(42)
df = pd.read_csv("data/data_full.csv")

FEATURES_4 = ["depth_ft", "slope", "soil_shear_lbft2", "velocity_ms"]
FEATURES_6 = ["depth_ft", "slope", "soil_shear_lbft2", "velocity_ms", "breadth_m", "channel_avg_depth_m"]
TARGET = "sed_transport_rate_kgsm"
y = df[TARGET].values

def mlp_builder():
    return MLPRegressor(hidden_layer_sizes=(8, 4), activation='relu', solver='lbfgs',
                         alpha=0.1, max_iter=5000, random_state=42)

def lin_builder():
    return LinearRegression()

def evaluate_raw(feature_list, model_builder, n_splits=5, n_repeats=10, seed=42):
    X = df[feature_list].values
    kf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    rows = []
    for fold_i, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        scaler = StandardScaler().fit(X_train)
        X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)
        model = model_builder()
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)
        rows.append({
            "fold": fold_i,
            "repeat": fold_i // n_splits,
            "r2": r2_score(y_test, pred),
            "rmse": np.sqrt(mean_squared_error(y_test, pred)),
            "mae": mean_absolute_error(y_test, pred),
        })
    return pd.DataFrame(rows)

results = {}
raw_frames = {}
for name, feats, builder in [
    ("mlp_6feat", FEATURES_6, mlp_builder),
    ("mlp_4feat", FEATURES_4, mlp_builder),
    ("linear_6feat", FEATURES_6, lin_builder),
    ("linear_4feat", FEATURES_4, lin_builder),
]:
    raw = evaluate_raw(feats, builder)
    raw_frames[name] = raw
    raw.to_csv(f"results/cv_raw_{name}.csv", index=False)
    results[name] = {
        "r2_mean": float(raw.r2.mean()), "r2_std": float(raw.r2.std()),
        "rmse_mean": float(raw.rmse.mean()), "rmse_std": float(raw.rmse.std()),
        "mae_mean": float(raw.mae.mean()), "mae_std": float(raw.mae.std()),
    }

with open("results/cv_results_all.json", "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))

# ---- Hold-out for 6-feature and 4-feature models (same split) ----
def holdout_eval(feature_list, model_builder, seed=7):
    X = df[feature_list].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)
    model = model_builder()
    model.fit(X_train_s, y_train)
    pred = model.predict(X_test_s)
    metrics = {
        "r2": float(r2_score(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "mae": float(mean_absolute_error(y_test, pred)),
    }
    return model, scaler, X_test_s, y_test, pred, metrics

mlp6_model, mlp6_scaler, X_test6_s, y_test6, pred6, holdout6 = holdout_eval(FEATURES_6, mlp_builder)
mlp4_model, mlp4_scaler, X_test4_s, y_test4, pred4, holdout4 = holdout_eval(FEATURES_4, mlp_builder)
lin6_model, lin6_scaler, Xl_test6_s, yl_test6, predl6, holdout_lin6 = holdout_eval(FEATURES_6, lin_builder)

print("Hold-out 6-feat MLP:", holdout6)
print("Hold-out 4-feat MLP:", holdout4)
print("Hold-out 6-feat Linear:", holdout_lin6)

pd.DataFrame({"observed": y_test6, "predicted_mlp6": np.round(pred6, 3),
              "predicted_linear6": np.round(predl6, 3)}).to_csv("results/holdout_compare.csv", index=False)

with open("results/holdout_all_metrics.json", "w") as f:
    json.dump({"mlp_6feat": holdout6, "mlp_4feat": holdout4, "linear_6feat": holdout_lin6}, f, indent=2)

# ---- Permutation importance: 6-feature and 4-feature models ----
perm6 = permutation_importance(mlp6_model, X_test6_s, y_test6, n_repeats=30, random_state=42, scoring="r2")
imp6 = {feat: {"mean": float(m), "std": float(s)} for feat, m, s in
        zip(FEATURES_6, perm6.importances_mean, perm6.importances_std)}

perm4 = permutation_importance(mlp4_model, X_test4_s, y_test4, n_repeats=30, random_state=42, scoring="r2")
imp4 = {feat: {"mean": float(m), "std": float(s)} for feat, m, s in
        zip(FEATURES_4, perm4.importances_mean, perm4.importances_std)}

with open("results/importance_6feat.json", "w") as f:
    json.dump(imp6, f, indent=2)
with open("results/importance_4feat.json", "w") as f:
    json.dump(imp4, f, indent=2)

print("6-feat importance:", json.dumps(imp6, indent=2))
print("4-feat importance:", json.dumps(imp4, indent=2))

print("\nAll analysis artifacts written.")
