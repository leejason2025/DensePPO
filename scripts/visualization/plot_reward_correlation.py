"""Regenerate cleaner reward correlation plot from saved raw results."""
import json, numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

OUT_DIR  = "/234/outputs/plots"
JSON_PATH = "/234/outputs/plots/reward_correlation_raw.json"

with open(JSON_PATH) as f:
    raw = json.load(f)

# json keys are strings
all_results = {int(k): v for k, v in raw.items()}

# ── Filter tasks with both successes and failures ─────────────────────────────
TASK_SHORT = {
    0: "T0: Between\nplate & ramekin",
    1: "T1: Next to\nramekin",
    2: "T2: Table center",
    3: "T3: On\ncookie box",
    4: "T4: Top drawer",
    5: "T5: On ramekin",
    6: "T6: Next to\ncookie box",
    7: "T7: On stove",
    8: "T8: Next to plate",
    9: "T9: On wooden\ncabinet",
}

valid_tasks = []
for task_id, results in all_results.items():
    success_dr = [r["cum_dense"] for r in results if r["success"]]
    fail_dr    = [r["cum_dense"] for r in results if not r["success"]]
    if success_dr and fail_dr:
        valid_tasks.append(task_id)

print(f"Tasks with both success and failure: {valid_tasks}")

# ── Main plot: clean boxplot for valid tasks only ─────────────────────────────
n = len(valid_tasks)
ncols = 4
nrows = (n + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
axes = axes.flatten()

separations = []

for idx, task_id in enumerate(valid_tasks):
    ax = axes[idx]
    results    = all_results[task_id]
    success_dr = [r["cum_dense"] for r in results if r["success"]]
    fail_dr    = [r["cum_dense"] for r in results if not r["success"]]

    bp = ax.boxplot(
        [fail_dr, success_dr],
        labels=["Failure", "Success"],
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker="o", markersize=5, alpha=0.5),
    )
    bp["boxes"][0].set_facecolor("#e74c3c")
    bp["boxes"][0].set_alpha(0.75)
    bp["boxes"][1].set_facecolor("#2ecc71")
    bp["boxes"][1].set_alpha(0.75)

    # t-test
    t, p = stats.ttest_ind(success_dr, fail_dr)
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))

    sep = np.mean(success_dr) - np.mean(fail_dr)
    separations.append(sep)

    ax.set_title(f"{TASK_SHORT[task_id]}\n"
                 f"n_success={len(success_dr)}, n_fail={len(fail_dr)}  {sig}",
                 fontsize=9)
    ax.set_ylabel("Cumulative Dense Reward", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # annotate means
    ax.text(1, np.mean(fail_dr),    f"μ={np.mean(fail_dr):.2f}",
            ha="center", va="bottom", fontsize=8, color="#c0392b")
    ax.text(2, np.mean(success_dr), f"μ={np.mean(success_dr):.2f}",
            ha="center", va="bottom", fontsize=8, color="#27ae60")

# hide unused axes
for idx in range(len(valid_tasks), len(axes)):
    axes[idx].set_visible(False)

mean_sep = np.mean(separations)
fig.suptitle(
    f"Cumulative Dense Reward: Success vs Failure Episodes\n"
    f"Mean separation across tasks = {mean_sep:.3f}  "
    f"(* p<0.05, ** p<0.01, *** p<0.001, ns = not significant)",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/reward_correlation_clean.png", dpi=150, bbox_inches="tight")
print(f"Saved reward_correlation_clean.png")


# ── Summary bar chart: mean reward gap per task ───────────────────────────────
fig2, ax = plt.subplots(figsize=(10, 4))

task_labels = [TASK_SHORT[t].replace("\n", " ") for t in valid_tasks]
colors = ["#2ecc71" if s > 0 else "#e74c3c" for s in separations]
bars = ax.bar(range(len(valid_tasks)), separations, color=colors, alpha=0.8, edgecolor="black", linewidth=0.8)

ax.axhline(0, color="black", linewidth=1)
ax.axhline(mean_sep, color="#3498db", linewidth=2, linestyle="--",
           label=f"Mean separation = {mean_sep:.3f}")
ax.set_xticks(range(len(valid_tasks)))
ax.set_xticklabels(task_labels, fontsize=9, rotation=15, ha="right")
ax.set_ylabel("Mean Reward (Success) − Mean Reward (Failure)", fontsize=10)
ax.set_title("Dense Reward Separation Between Success and Failure Episodes\n"
             "(positive = reward correctly ranks successful episodes higher)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)

# value labels on bars
for bar, val in zip(bars, separations):
    ax.text(bar.get_x() + bar.get_width()/2,
            val + (0.005 if val >= 0 else -0.015),
            f"{val:.2f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/reward_separation_bar.png", dpi=150, bbox_inches="tight")
print(f"Saved reward_separation_bar.png")

print(f"\nMean reward separation (success - failure): {mean_sep:.3f}")
for task_id, sep in zip(valid_tasks, separations):
    print(f"  Task {task_id}: {sep:+.3f}")
