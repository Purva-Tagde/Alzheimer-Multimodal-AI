
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Output location
output_dir = Path("reports")
output_dir.mkdir(exist_ok=True)

# Verified ADNI model results (percentages)
models = ["Logistic Regression", "Random Forest", "XGBoost"]

metrics = {
    "Accuracy":    [80.30, 87.88, 84.85],
    "Precision":   [79.17, 85.19, 84.00],
    "Recall":      [70.37, 85.19, 77.78],
    "Specificity": [87.18, 89.74, 89.74],
    "F1-score":    [74.51, 85.19, 80.77],
}

x = np.arange(len(metrics))
width = 0.24

fig, ax = plt.subplots(figsize=(12, 6.5))

for i, model in enumerate(models):
    values = [metrics[metric][i] for metric in metrics]
    bars = ax.bar(
        x + (i - 1) * width,
        values,
        width,
        label=model
    )
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)

ax.set_title("ADNI: Model Performance Comparison", fontsize=15, weight="bold")
ax.set_ylabel("Score (%)")
ax.set_xticks(x)
ax.set_xticklabels(list(metrics.keys()))
ax.set_ylim(0, 105)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3)
ax.grid(axis="y", linestyle="--", alpha=0.35)
ax.set_axisbelow(True)

fig.tight_layout()

output_path = output_dir / "adni_model_comparison.png"
fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"Graph saved to: {output_path.resolve()}")
