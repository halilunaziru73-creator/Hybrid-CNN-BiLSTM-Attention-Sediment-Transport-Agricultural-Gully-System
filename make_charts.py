import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import json

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold"})

df = pd.read_csv("data/data_full.csv")
FEATURES_6 = ["depth_ft", "slope", "soil_shear_lbft2", "velocity_ms", "breadth_m", "channel_avg_depth_m"]
FEATURE_LABELS = {
    "depth_ft": "Flow depth (ft)", "slope": "Channel slope (-)",
    "soil_shear_lbft2": "Soil shear (lb/ft\u00B2)", "velocity_ms": "Flow velocity (m/s)",
    "breadth_m": "Channel breadth (m)", "channel_avg_depth_m": "Channel avg. depth (m)",
}
TARGET = "sed_transport_rate_kgsm"
COLOR = "#2E5A88"
COLOR2 = "#C0562D"
COLOR3 = "#4E9B6E"
COLOR4 = "#8B6BAF"

# ============ 3.1 EDA charts ============

# 1. Histogram of target
plt.figure(figsize=(5.5, 4))
plt.hist(df[TARGET], bins=15, color=COLOR, edgecolor="white")
plt.xlabel("Sediment transport rate (kg/s/m)")
plt.ylabel("Frequency")
plt.title("Distribution of sediment transport rate (n = 100)")
plt.tight_layout()
plt.savefig("figures/fig_hist_target.png", dpi=200)
plt.close()

# 2. Correlation heatmap
corr = df[FEATURES_6 + [TARGET]].corr()
labels = [FEATURE_LABELS.get(c, c) for c in FEATURES_6] + ["Sediment transport rate"]
plt.figure(figsize=(6.2, 5.5))
im = plt.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
plt.yticks(range(len(labels)), labels)
for i in range(len(labels)):
    for j in range(len(labels)):
        plt.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center",
                  color="white" if abs(corr.values[i,j]) > 0.6 else "black", fontsize=8)
plt.colorbar(im, fraction=0.046, pad=0.04, label="Pearson correlation")
plt.title("Correlation matrix of model variables")
plt.tight_layout()
plt.savefig("figures/fig_corr_heatmap.png", dpi=200)
plt.close()

# 3. Feature scatter grid (each feature vs target)
fig, axes = plt.subplots(2, 3, figsize=(9.5, 6))
for ax, feat in zip(axes.flat, FEATURES_6):
    ax.scatter(df[feat], df[TARGET], s=18, color=COLOR, alpha=0.75, edgecolor="white", linewidth=0.4)
    ax.set_xlabel(FEATURE_LABELS[feat], fontsize=9)
    ax.set_ylabel("Sediment transport\nrate (kg/s/m)", fontsize=8)
    r = np.corrcoef(df[feat], df[TARGET])[0, 1]
    ax.set_title(f"r = {r:.2f}", fontsize=9)
fig.suptitle("Sediment transport rate versus each predictor variable", fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("figures/fig_feature_scatter_grid.png", dpi=200)
plt.close()

# 4. Boxplot by scenario
scenario_order = ["pre_control", "pond10_pre", "pond10_post", "pond15_pre", "pond15_post"]
scenario_labels = ["Pre-control", "1.0 m pre-\ncontrol", "1.0 m post-\ncontrol", "1.5 m pre-\ncontrol", "1.5 m post-\ncontrol"]
data_by_scenario = [df[df.scenario == s][TARGET].values for s in scenario_order]
plt.figure(figsize=(6.5, 4.5))
bp = plt.boxplot(data_by_scenario, labels=scenario_labels, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor("#AFC6E3")
plt.ylabel("Sediment transport rate (kg/s/m)")
plt.title("Sediment transport rate by field scenario")
plt.xticks(fontsize=8.5)
plt.tight_layout()
plt.savefig("figures/fig_boxplot_scenario.png", dpi=200)
plt.close()

# ============ 3.2 CV performance charts ============
raw = {name: pd.read_csv(f"results/cv_raw_{name}.csv") for name in
       ["mlp_6feat", "mlp_4feat", "linear_6feat", "linear_4feat"]}
model_labels = {"mlp_6feat": "DL, 6 var", "mlp_4feat": "DL, 4 var",
                "linear_6feat": "Linear, 6 var", "linear_4feat": "Linear, 4 var"}
order = ["mlp_6feat", "mlp_4feat", "linear_6feat", "linear_4feat"]
colors = [COLOR, COLOR3, COLOR2, COLOR4]

# 5. CV R2 bar with error bars
means = [raw[m].r2.mean() for m in order]
stds = [raw[m].r2.std() for m in order]
plt.figure(figsize=(6, 4.3))
plt.bar([model_labels[m] for m in order], means, yerr=stds, capsize=6, color=colors, edgecolor="black", linewidth=0.6)
plt.ylabel("R\u00B2 (mean \u00B1 SD, 50 folds)")
plt.title("Cross-validated R\u00B2 by model")
plt.axhline(0, color="grey", linewidth=0.8)
plt.tight_layout()
plt.savefig("figures/fig_cv_r2_bar.png", dpi=200)
plt.close()

# 6. CV R2 boxplot (fold distributions)
plt.figure(figsize=(6.5, 4.5))
bp = plt.boxplot([raw[m].r2 for m in order], labels=[model_labels[m] for m in order], patch_artist=True)
for patch, c in zip(bp['boxes'], colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.75)
plt.ylabel("R\u00B2 (50 folds)")
plt.title("Distribution of fold-wise R\u00B2 across models")
plt.tight_layout()
plt.savefig("figures/fig_cv_r2_boxplot.png", dpi=200)
plt.close()

# 7. Grouped bar RMSE and MAE
x = np.arange(len(order))
width = 0.35
rmse_means = [raw[m].rmse.mean() for m in order]
mae_means = [raw[m].mae.mean() for m in order]
plt.figure(figsize=(6.5, 4.5))
plt.bar(x - width/2, rmse_means, width, label="RMSE", color=COLOR, edgecolor="black", linewidth=0.5)
plt.bar(x + width/2, mae_means, width, label="MAE", color=COLOR2, edgecolor="black", linewidth=0.5)
plt.xticks(x, [model_labels[m] for m in order])
plt.ylabel("Error (kg/s/m)")
plt.title("Cross-validated RMSE and MAE by model")
plt.legend()
plt.tight_layout()
plt.savefig("figures/fig_cv_rmse_mae_bar.png", dpi=200)
plt.close()

# 8. Fold-wise (per-repeat mean) line plot for mlp6 vs linear6
rep_mlp6 = raw["mlp_6feat"].groupby("repeat").r2.mean()
rep_lin6 = raw["linear_6feat"].groupby("repeat").r2.mean()
plt.figure(figsize=(6.5, 4.3))
plt.plot(rep_mlp6.index + 1, rep_mlp6.values, marker="o", color=COLOR, label="DL, 6 var")
plt.plot(rep_lin6.index + 1, rep_lin6.values, marker="s", color=COLOR2, label="Linear, 6 var")
plt.xlabel("Cross-validation repeat (1-10)")
plt.ylabel("Mean R\u00B2 across 5 folds")
plt.title("Repeat-wise mean R\u00B2: deep learning vs. linear regression")
plt.legend()
plt.xticks(range(1, 11))
plt.tight_layout()
plt.savefig("figures/fig_cv_foldwise_line.png", dpi=200)
plt.close()

# ============ 3.3 Hold-out charts ============
hc = pd.read_csv("results/holdout_compare.csv")

# 9. Predicted vs observed (6-feat MLP) -- reuse/regenerate
plt.figure(figsize=(5.3, 5))
mx = max(hc.observed.max(), hc.predicted_mlp6.max()) * 1.1
plt.scatter(hc.observed, hc.predicted_mlp6, color=COLOR, s=55, edgecolor="white", zorder=3)
plt.plot([0, mx], [0, mx], linestyle="--", color="#999999", linewidth=1.2, label="1:1 line")
plt.xlabel("Observed sediment transport rate (kg/s/m)")
plt.ylabel("Predicted sediment transport rate (kg/s/m)")
plt.title("Deep learning model (6 variables):\npredicted vs. observed (n = 20)")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("figures/fig_holdout_pred_obs_6feat.png", dpi=200)
plt.close()

# 10. Residual plot
residuals = hc.predicted_mlp6 - hc.observed
plt.figure(figsize=(5.8, 4.3))
plt.axhline(0, color="grey", linewidth=1)
plt.scatter(hc.observed, residuals, color=COLOR, s=50, edgecolor="white")
plt.xlabel("Observed sediment transport rate (kg/s/m)")
plt.ylabel("Residual (predicted \u2212 observed)")
plt.title("Residuals of the deep learning model on the hold-out set")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("figures/fig_holdout_residual.png", dpi=200)
plt.close()

# 11. Error histogram
abs_err = residuals.abs()
plt.figure(figsize=(5.5, 4.2))
plt.hist(abs_err, bins=8, color=COLOR3, edgecolor="white")
plt.xlabel("Absolute error (kg/s/m)")
plt.ylabel("Frequency")
plt.title("Distribution of absolute prediction errors (hold-out set)")
plt.tight_layout()
plt.savefig("figures/fig_holdout_error_hist.png", dpi=200)
plt.close()

# 12. Index-ordered actual vs predicted line
order_idx = np.argsort(-hc.observed.values)
plt.figure(figsize=(6.5, 4.3))
plt.plot(range(1, 21), hc.observed.values[order_idx], marker="o", color="black", label="Observed")
plt.plot(range(1, 21), hc.predicted_mlp6.values[order_idx], marker="^", color=COLOR, label="Predicted (DL, 6 var)")
plt.xlabel("Hold-out test observation (ranked by observed value)")
plt.ylabel("Sediment transport rate (kg/s/m)")
plt.title("Observed vs. predicted sediment transport rate, ranked")
plt.legend()
plt.tight_layout()
plt.savefig("figures/fig_holdout_index_line.png", dpi=200)
plt.close()

# ============ 3.4 Importance charts ============
with open("results/importance_6feat.json") as f:
    imp6 = json.load(f)
with open("results/importance_4feat.json") as f:
    imp4 = json.load(f)

feats6_sorted = sorted(imp6.items(), key=lambda kv: kv[1]["mean"], reverse=True)
names6 = [FEATURE_LABELS[k] for k, v in feats6_sorted]
means6 = [v["mean"] for k, v in feats6_sorted]
stds6 = [v["std"] for k, v in feats6_sorted]

# 13. Importance bar with error bars
plt.figure(figsize=(6.5, 4.5))
plt.barh(names6[::-1], means6[::-1], xerr=stds6[::-1], color=COLOR, edgecolor="black", capsize=4)
plt.xlabel("Mean R\u00B2 drop when variable is permuted (30 repeats)")
plt.title("Permutation importance, six-variable deep learning model")
plt.tight_layout()
plt.savefig("figures/fig_importance_bar.png", dpi=200)
plt.close()

# 14. 4-var vs 6-var importance comparison (common features)
common = [f for f in FEATURES_6 if f in imp4]
x = np.arange(len(common))
width = 0.35
m6 = [imp6[f]["mean"] for f in common]
m4 = [imp4[f]["mean"] for f in common]
plt.figure(figsize=(6.5, 4.5))
plt.bar(x - width/2, m4, width, label="4-variable model", color=COLOR3, edgecolor="black")
plt.bar(x + width/2, m6, width, label="6-variable model", color=COLOR, edgecolor="black")
plt.xticks(x, [FEATURE_LABELS[f] for f in common], rotation=30, ha="right", fontsize=8.5)
plt.ylabel("Mean R\u00B2 drop when permuted")
plt.title("Importance of shared variables: 4-variable vs. 6-variable model")
plt.legend()
plt.tight_layout()
plt.savefig("figures/fig_importance_4v6.png", dpi=200)
plt.close()

# 15. Pareto / cumulative importance (6-var, only positive contributions)
pos = [(n, m) for n, m in zip(names6, means6) if m > 0]
names_p = [n for n, m in pos]
means_p = [m for n, m in pos]
cum = np.cumsum(means_p) / np.sum(means_p) * 100
fig, ax1 = plt.subplots(figsize=(6.5, 4.5))
ax1.bar(names_p, means_p, color=COLOR4, edgecolor="black")
ax1.set_ylabel("Mean R\u00B2 drop")
ax1.set_xticklabels(names_p, rotation=30, ha="right", fontsize=8.5)
ax2 = ax1.twinx()
ax2.plot(names_p, cum, color=COLOR2, marker="o")
ax2.set_ylabel("Cumulative share (%)")
ax2.set_ylim(0, 110)
plt.title("Pareto chart of predictor importance (6-variable model)")
plt.tight_layout()
plt.savefig("figures/fig_importance_pareto.png", dpi=200)
plt.close()

# ============ 3.5 Comparison charts ============
with open("results/cv_results_all.json") as f:
    cv_all = json.load(f)
with open("results/holdout_all_metrics.json") as f:
    ho_all = json.load(f)

# 16. Grouped bar: CV R2, RMSE, MAE across all 4 models (normalized subplot layout)
fig, axes = plt.subplots(1, 3, figsize=(11, 4))
metrics = ["r2_mean", "rmse_mean", "mae_mean"]
titles = ["R\u00B2 (higher is better)", "RMSE, kg/s/m (lower is better)", "MAE, kg/s/m (lower is better)"]
for ax, metric, title in zip(axes, metrics, titles):
    vals = [cv_all[m][metric] for m in order]
    ax.bar([model_labels[m] for m in order], vals, color=colors, edgecolor="black")
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis='x', labelrotation=25, labelsize=8)
fig.suptitle("Summary comparison of cross-validated performance across all models", fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("figures/fig_grouped_bar_all_models.png", dpi=200)
plt.close()

# 17. Radar chart (normalized: R2 as-is, 1/RMSE, 1/MAE scaled 0-1)
def norm(vals, invert=False):
    vals = np.array(vals, dtype=float)
    if invert:
        vals = 1.0 / vals
    v_min, v_max = vals.min(), vals.max()
    return (vals - v_min) / (v_max - v_min + 1e-9)

cats = ["R\u00B2", "1/RMSE", "1/MAE"]
r2v = [cv_all[m]["r2_mean"] for m in order]
rmsev = [cv_all[m]["rmse_mean"] for m in order]
maev = [cv_all[m]["mae_mean"] for m in order]
r2n, rmsen, maen = norm(r2v), norm(rmsev, invert=True), norm(maev, invert=True)

angles = np.linspace(0, 2*np.pi, len(cats), endpoint=False).tolist()
angles += angles[:1]
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, polar=True)
for i, m in enumerate(order):
    vals = [r2n[i], rmsen[i], maen[i]]
    vals += vals[:1]
    ax.plot(angles, vals, label=model_labels[m], color=colors[i], linewidth=1.8)
    ax.fill(angles, vals, color=colors[i], alpha=0.08)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(cats)
ax.set_yticklabels([])
plt.title("Normalised cross-validated performance profile\n(outer = better)", fontsize=10, y=1.08)
plt.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
plt.tight_layout()
plt.savefig("figures/fig_metrics_radar.png", dpi=200)
plt.close()

# 18. Scatter overlay: DL vs Linear predictions vs observed (hold-out)
plt.figure(figsize=(5.6, 5.2))
mx = max(hc.observed.max(), hc.predicted_mlp6.max(), hc.predicted_linear6.max()) * 1.1
plt.plot([0, mx], [0, mx], linestyle="--", color="#999999", linewidth=1.2, label="1:1 line")
plt.scatter(hc.observed, hc.predicted_mlp6, color=COLOR, s=50, edgecolor="white", label="Deep learning (6 var)")
plt.scatter(hc.observed, hc.predicted_linear6, color=COLOR2, s=50, marker="^", edgecolor="white", label="Linear regression (6 var)")
plt.xlabel("Observed sediment transport rate (kg/s/m)")
plt.ylabel("Predicted sediment transport rate (kg/s/m)")
plt.title("Hold-out predictions: deep learning vs. linear regression")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("figures/fig_comparison_scatter_overlay.png", dpi=200)
plt.close()

# ============ Methods: network architecture diagram ============
fig, ax = plt.subplots(figsize=(7.5, 4.2))
layer_sizes = [6, 8, 4, 1]
layer_names = ["Input\n(6 variables)", "Hidden layer 1\n(8 neurons, ReLU)", "Hidden layer 2\n(4 neurons, ReLU)", "Output\n(1 neuron, linear)"]
layer_x = [0.5, 2.2, 3.9, 5.6]
v_positions = []
max_n = max(layer_sizes)
for x, n in zip(layer_x, layer_sizes):
    ys = np.linspace(0.5, max_n - 0.5, n) if n > 1 else [max_n/2]
    ys = np.array(ys) + (max_n - max(ys) - min(ys)) / 2 if n > 1 else ys
    v_positions.append(ys)

for li in range(len(layer_x) - 1):
    for y1 in v_positions[li]:
        for y2 in v_positions[li + 1]:
            ax.plot([layer_x[li], layer_x[li+1]], [y1, y2], color="#CCCCCC", linewidth=0.4, zorder=1)

node_colors = ["#4E9B6E", "#2E5A88", "#2E5A88", "#C0562D"]
for x, ys, c in zip(layer_x, v_positions, node_colors):
    for y in ys:
        circ = plt.Circle((x, y), 0.22, color=c, ec="black", linewidth=0.6, zorder=2)
        ax.add_patch(circ)

for x, name, n in zip(layer_x, layer_names, layer_sizes):
    ax.text(x, max_n + 0.4, name, ha="center", fontsize=8.5, fontweight="bold")

ax.set_xlim(-0.3, 6.4)
ax.set_ylim(-0.3, max_n + 1.1)
ax.axis("off")
ax.set_title("Feedforward multilayer perceptron architecture used for\nsediment transport rate prediction", fontsize=10.5, fontweight="bold")
plt.tight_layout()
plt.savefig("figures/fig_network_architecture.png", dpi=200)
plt.close()

print("All figures generated.")
import subprocess
print(subprocess.run(["ls", "-1"], capture_output=True, text=True).stdout)
