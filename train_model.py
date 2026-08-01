import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split, RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json

np.random.seed(42)

df = pd.read_csv("data/data.csv")
print("Dataset shape:", df.shape)
print(df.describe())

# Features: depth, slope, soil shear, velocity  -> Target: sediment transport rate (kg/s/m)
X = df[["depth_ft", "slope", "soil_shear_lbft2", "velocity_ms"]].values
y = df["sed_transport_rate_kgsm"].values

# ---- 5-fold cross-validation comparison: MLP (deep learning) vs Linear Regression (thesis baseline) ----
kf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)

mlp_r2, mlp_rmse, mlp_mae = [], [], []
lin_r2, lin_rmse, lin_mae = [], [], []

for train_idx, test_idx in kf.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Deep learning model: MLP with two hidden layers
    mlp = MLPRegressor(hidden_layer_sizes=(8, 4), activation='relu',
                        solver='lbfgs', alpha=0.1, max_iter=5000,
                        random_state=42)
    mlp.fit(X_train_s, y_train)
    pred = mlp.predict(X_test_s)
    mlp_r2.append(r2_score(y_test, pred))
    mlp_rmse.append(np.sqrt(mean_squared_error(y_test, pred)))
    mlp_mae.append(mean_absolute_error(y_test, pred))

    # Baseline: simple linear regression (same style as thesis empirical model, but multivariate)
    lin = LinearRegression()
    lin.fit(X_train_s, y_train)
    pred_l = lin.predict(X_test_s)
    lin_r2.append(r2_score(y_test, pred_l))
    lin_rmse.append(np.sqrt(mean_squared_error(y_test, pred_l)))
    lin_mae.append(mean_absolute_error(y_test, pred_l))

results = {
    "mlp": {
        "r2_mean": float(np.mean(mlp_r2)), "r2_std": float(np.std(mlp_r2)),
        "rmse_mean": float(np.mean(mlp_rmse)), "rmse_std": float(np.std(mlp_rmse)),
        "mae_mean": float(np.mean(mlp_mae)), "mae_std": float(np.std(mlp_mae)),
    },
    "linear": {
        "r2_mean": float(np.mean(lin_r2)), "r2_std": float(np.std(lin_r2)),
        "rmse_mean": float(np.mean(lin_rmse)), "rmse_std": float(np.std(lin_rmse)),
        "mae_mean": float(np.mean(lin_mae)), "mae_std": float(np.std(lin_mae)),
    }
}
print(json.dumps(results, indent=2))

# Also fit thesis-style simple 1-variable linear model: Qs vs velocity only (to mirror thesis Eq 4.1)
X_v = df[["velocity_ms"]].values
lin_simple = LinearRegression().fit(X_v, y)
print("\nSimple velocity-only linear model (mirrors thesis Eq 4.1):")
print("slope:", lin_simple.coef_, "intercept:", lin_simple.intercept_)
pred_simple = lin_simple.predict(X_v)
print("R2 (in-sample, whole dataset):", r2_score(y, pred_simple))

# ---- Final model: train on full dataset (80/20 hold-out) for reporting in the paper ----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

final_mlp = MLPRegressor(hidden_layer_sizes=(8, 4), activation='relu', solver='lbfgs',
                          alpha=0.1, max_iter=5000, random_state=42)
final_mlp.fit(X_train_s, y_train)
pred_test = final_mlp.predict(X_test_s)
print("\nHold-out test set (n=%d):" % len(y_test))
print("R2:", r2_score(y_test, pred_test))
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred_test)))
print("MAE:", mean_absolute_error(y_test, pred_test))

# Save predictions vs observed for a table in the paper
out = pd.DataFrame({
    "observed_kg_s_m": y_test,
    "predicted_kg_s_m": np.round(pred_test, 3)
})
out.to_csv("results/holdout_predictions.csv", index=False)
print(out)

with open("results/cv_results.json", "w") as f:
    json.dump(results, f, indent=2)
