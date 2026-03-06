import os, numpy as np, matplotlib.pyplot as plt, matplotlib.cm as cm, torch

exp_dir = "/234/LIBERO/experiments/LIBERO_SPATIAL/Sequential/BCTransformerPolicy_seed42/run_003"

TASK_NAMES = [
    "Between plate & ramekin", "Next to ramekin", "Table center",
    "On cookie box", "Top drawer", "On ramekin",
    "Next to cookie box", "On stove", "Next to plate", "On wooden cabinet",
]

fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharey=False)
axes = axes.flatten()
colors = cm.tab10(np.linspace(0, 1, 10))

for i in range(10):
    log_path = os.path.join(exp_dir, f"task{i}_auc.log")
    ax = axes[i]
    if not os.path.exists(log_path):
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"Task {i}", fontsize=9)
        continue
    data = torch.load(log_path)
    loss = data["loss"]
    epochs = np.arange(len(loss)) * 5

    # shift so all values are positive before log
    shifted = loss - loss.min() + 1e-3
    ax.plot(epochs, np.log(shifted), color=colors[i], linewidth=2, marker="o", markersize=4)
    ax.set_title(f"Task {i}: {TASK_NAMES[i]}\nFinal: {loss[-1]:.2f}", fontsize=8)
    ax.set_xlabel("Epoch", fontsize=8)
    ax.set_ylabel("Log Loss (shifted)", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle("LIBERO-Spatial: BCTransformer Training Loss (Log Scale)",
    fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
out = os.path.join(exp_dir, "training_loss_log.png")
plt.savefig(out, bbox_inches="tight", dpi=150)
print(f"Saved to {out}")
