"""Grouped bar chart: BC vs BC+Sparse PPO vs BC+Dense PPO per task."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT_DIR = "/234/outputs/plots"

bc     = [80, 80, 100, 90, 75, 70, 85, 90, 80, 65]
sparse = [80, 95, 100, 85, 75, 90, 85, 95, 70, 80]
dense  = [75, 95,  95, 90,100, 85, 90, 95, 85, 75]

bc_avg     = np.mean(bc)
sparse_avg = np.mean(sparse)
dense_avg  = np.mean(dense)

tasks = [f"T{i}" for i in range(10)]
x     = np.arange(len(tasks))
w     = 0.25

BC_COLOR     = "#95a5a6"
SPARSE_COLOR = "#e67e22"
DENSE_COLOR  = "#2980b9"

fig, ax = plt.subplots(figsize=(13, 6))

b1 = ax.bar(x - w,   bc,     w, label=f"BC Baseline (avg={bc_avg:.1f}%)",        color=BC_COLOR,     edgecolor="white", linewidth=0.5)
b2 = ax.bar(x,       sparse, w, label=f"BC + Sparse PPO (avg={sparse_avg:.1f}%)", color=SPARSE_COLOR, edgecolor="white", linewidth=0.5)
b3 = ax.bar(x + w,   dense,  w, label=f"BC + Dense PPO (avg={dense_avg:.1f}%)",  color=DENSE_COLOR,  edgecolor="white", linewidth=0.5)

# value labels on top of bars
for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                f"{int(h)}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

# avg lines
ax.axhline(bc_avg,     color=BC_COLOR,     linewidth=1.5, linestyle="--", alpha=0.7)
ax.axhline(sparse_avg, color=SPARSE_COLOR, linewidth=1.5, linestyle="--", alpha=0.7)
ax.axhline(dense_avg,  color=DENSE_COLOR,  linewidth=1.5, linestyle="--", alpha=0.7)

ax.set_xticks(x)
ax.set_xticklabels(tasks, fontsize=11)
ax.set_ylabel("Success Rate (%)", fontsize=12)
ax.set_ylim(0, 115)
ax.set_xlabel("Task", fontsize=12)
ax.set_title("Per-Task Success Rates: BC Baseline vs BC + Sparse PPO vs BC + Dense PPO\n"
             "Dashed lines = average across all 10 tasks",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10, loc="upper right")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/success_comparison.png", dpi=150, bbox_inches="tight")
print("Saved success_comparison.png")
plt.close()
