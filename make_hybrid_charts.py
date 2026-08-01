"""
Figures specific to the Hybrid CNN-BiLSTM-Attention model (paper Figures
10-11): hold-out predicted-vs-observed, CV comparison across all three
6-variable models, permutation importance, Kernel SHAP summary, and
Monte Carlo dropout prediction intervals.

Run after run_hybrid_analysis.py has populated results/hybrid_*.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLOR = "#2E5A88"
COLOR2 = "#C0562D"
COLOR3 = "#4E9B6E"
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold"})

FEATURES_6 = ["depth_ft", "slope", "soil_shear_lbft2", "velocity_ms",
              "breadth_m", "channel_avg_depth_m"]
FEATURE_LABELS = {
    "depth_ft": "Flow depth (ft)", "slope": "Channel slope (-)",
    "soil_shear_lbft2": "Soil shear (lb/ft\u00b2)", "velocity_ms": "Flow velocity (m/s)",
    "breadth_m": "Channel breadth (m)", "channel_avg_depth_m": "Channel avg. depth (m)",
}

# ---- Figure 10: hold-out scatter + residuals + CV comparison ----
hc = pd.read_csv("results/hybrid_holdout_compare.csv")
hybrid_cv = pd.read_csv("results/hybrid_cv_raw.csv")
mlp_cv = pd.read_csv("results/cv_raw_mlp_6feat.csv")
lin_cv = pd.read_csv("results/cv_raw_linear_6feat.csv")

fig, axes = plt.subplots(1, 3, figsize=(13, 4.22))

ax = axes[0]
ax.scatter(hc.observed, hc.predicted, color=COLOR, s=40, alpha=0.8, edgecolor="white")
lims = [min(hc.observed.min(), hc.predicted.min()) - 0.2,
        max(hc.observed.max(), hc.predicted.max()) + 0.2]
ax.plot(lims, lims, "--", color="gray", linewidth=1)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("Observed (kg/s/m)"); ax.set_ylabel("Predicted (kg/s/m)")
ax.set_title("(A) Hold-out: predicted vs. observed")

ax = axes[1]
resid = hc.predicted - hc.observed
ax.scatter(hc.observed, resid, color=COLOR2, s=40, alpha=0.8, edgecolor="white")
ax.axhline(0, color="gray", linestyle="--", linewidth=1)
ax.set_xlabel("Observed (kg/s/m)"); ax.set_ylabel("Residual (predicted - observed)")
ax.set_title("(B) Residuals vs. observed")

ax = axes[2]
data_by_model = [lin_cv.r2, mlp_cv.r2, hybrid_cv.r2]
labels = ["Linear", "MLP", "Hybrid"]
bp = ax.boxplot(data_by_model, tick_labels=labels, patch_artist=True)
for patch, c in zip(bp["boxes"], [COLOR3, COLOR2, COLOR]):
    patch.set_facecolor(c); patch.set_alpha(0.6)
ax.set_ylabel("Cross-validated R\u00b2 (50 folds)")
ax.set_title("(C) CV R\u00b2: linear, MLP, hybrid")
ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)

plt.tight_layout()
plt.savefig("figures/fig_hybrid_holdout_and_cv.png", dpi=200)
plt.close()

# ---- Figure 11: importance, SHAP, MC dropout ----
with open("results/hybrid_importance.json") as f:
    importance = json.load(f)
with open("results/hybrid_shap_summary.json") as f:
    shap_summary = json.load(f)
shap_values = np.load("results/hybrid_shap_values.npy")
with open("results/hybrid_mc_dropout.json") as f:
    mc = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(14, 3.66))

ax = axes[0]
order = sorted(FEATURES_6, key=lambda f: -importance[f]["mean"])
means = [importance[f]["mean"] for f in order]
stds = [importance[f]["std"] for f in order]
labels = [FEATURE_LABELS[f] for f in order]
ax.barh(labels, means, xerr=stds, color=COLOR, alpha=0.85, capsize=3)
ax.invert_yaxis()
ax.set_xlabel("Mean R\u00b2 drop (30 repeats)")
ax.set_title("(A) Permutation importance")

ax = axes[1]
order_shap = sorted(FEATURES_6, key=lambda f: -shap_summary[f])
vals = [shap_summary[f] for f in order_shap]
labels_shap = [FEATURE_LABELS[f] for f in order_shap]
ax.barh(labels_shap, vals, color=COLOR3, alpha=0.85)
ax.invert_yaxis()
ax.set_xlabel("Mean |Kernel SHAP value|")
ax.set_title("(B) Kernel SHAP summary (n=15)")

ax = axes[2]
mean_pred = np.array(mc["mean"])
lo = np.array(mc["lower"])
hi = np.array(mc["upper"])
observed = hc.observed.values
order_idx = np.argsort(-observed)
x = np.arange(len(observed))
ax.errorbar(x, mean_pred[order_idx], yerr=[mean_pred[order_idx] - lo[order_idx],
            hi[order_idx] - mean_pred[order_idx]], fmt="o", color=COLOR,
            ecolor=COLOR, elinewidth=1.5, capsize=3, markersize=5, label="MC dropout 95% PI")
ax.scatter(x, observed[order_idx], color=COLOR2, marker="x", s=50, label="Observed", zorder=5)
ax.set_xlabel("Hold-out observation (ranked by magnitude)")
ax.set_ylabel("Sediment transport rate (kg/s/m)")
ax.set_title(f"(C) MC dropout 95% PI (coverage={mc['empirical_coverage']*100:.0f}%)")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("figures/fig_hybrid_importance_shap_mcdropout.png", dpi=200)
plt.close()

print("Hybrid-model figures written to figures/fig_hybrid_*.png")
