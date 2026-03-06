import os, numpy as np, matplotlib.pyplot as plt, matplotlib.cm as cm, torch

exp_dir = "/234/LIBERO/experiments/LIBERO_SPATIAL/Sequential/BCTransformerPolicy_seed42/run_003"

TASK_NAMES = [
    "Between plate & ramekin", "Next to ramekin", "Table center",
    "On cookie box", "Top drawer", "On ramekin",
    "Next to cookie box", "On stove", "Next to plate", "On wooden cabinet",
]

all_successes = []
all_losses = []
n_epochs = None

for i in range(10):
    log_path = os.path.join(exp_dir, f"task{i}_auc.log")
    data = torch.load(log_path)
    all_successes.append(data["success"])
    all_losses.append(data["loss"])
    n_epochs = len(data["success"])

epochs = np.arange(n_epochs) * 5
colors = cm.tab10(np.linspace(0, 1, 10))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- Left: Success Rate ---
ax = axes[0]
for i, (succ, name) in enumerate(zip(all_successes, TASK_NAMES)):
    ax.plot(epochs, succ * 100, color=colors[i], linewidth=1.2, alpha=0.6, label=name)

avg_succ = np.mean(all_successes, axis=0) * 100
ax.plot(epochs, avg_succ, color="black", linewidth=3, label="Average", zorder=10)

ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Success Rate (%)", fontsize=12)
ax.set_title("Success Rate vs Epoch\n(All Tasks)", fontsize=13, fontweight="bold")
ax.set_ylim(0, 105)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=7, loc="upper left", ncol=2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.annotate(f"Avg final: {avg_succ[-1]:.1f}%", xy=(epochs[-1], avg_succ[-1]),
    xytext=(-60, 10), textcoords="offset points", fontsize=9,
    arrowprops=dict(arrowstyle="->", color="black"), color="black")

# --- Right: Loss ---
ax2 = axes[1]
for i, (loss, name) in enumerate(zip(all_losses, TASK_NAMES)):
    ax2.plot(epochs, loss, color=colors[i], linewidth=1.2, alpha=0.6, label=name)

avg_loss = np.mean(all_losses, axis=0)
ax2.plot(epochs, avg_loss, color="black", linewidth=3, label="Average", zorder=10)

ax2.set_xlabel("Epoch", fontsize=12)
ax2.set_ylabel("Loss", fontsize=12)
ax2.set_title("Loss vs Epoch\n(All Tasks)", fontsize=13, fontweight="bold")
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=7, loc="upper right", ncol=2)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

avg_succ_best = np.mean([s.max() for s in all_successes]) * 100
fig.suptitle(f"LIBERO-Spatial: BCTransformer (ResNet-T) — Average Best Success: {avg_succ_best:.1f}%",
    fontsize=14, fontweight="bold", y=1.01)

plt.tight_layout()
out = os.path.join(exp_dir, "combined_curves.png")
plt.savefig(out, bbox_inches="tight", dpi=150)
print(f"Saved to {out}")
