"""
Waypoint Quality Visualization
- t-SNE of latent space colored by waypoint assignment (k=5)
- Timeline plot showing waypoint progression through demo trajectories
"""
import os, sys, numpy as np, torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

OUT_DIR     = "/234/outputs/waypoints"
LATENT_DIR  = "/234/outputs/waypoints"
K           = 5  # best k based on inertia analysis

SHORT_NAMES = [
    "Between plate & ramekin", "Next to ramekin", "Table center",
    "On cookie box", "Top drawer", "On ramekin",
    "Next to cookie box", "On stove", "Next to plate", "On wooden cabinet",
]

# ── 1. t-SNE grid: 2x5 subplots, one per task ────────────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()
colors = cm.tab10(np.linspace(0, 1, K))

for task_id in range(10):
    latents   = np.load(os.path.join(LATENT_DIR, f"task{task_id}_latents.npy"))
    results   = torch.load(os.path.join(LATENT_DIR, f"task{task_id}_waypoints.pt"))
    labels    = results[K]["labels"]   # (N,) cluster assignment per frame
    centers   = results[K]["centers"]  # (K, D)

    # subsample for speed if large
    if len(latents) > 2000:
        idx     = np.random.choice(len(latents), 2000, replace=False)
        latents_sub = latents[idx]
        labels_sub  = labels[idx]
    else:
        latents_sub = latents
        labels_sub  = labels

    latents_norm = normalize(latents_sub)

    print(f"Task {task_id}: running t-SNE on {len(latents_sub)} points...")
    tsne = TSNE(n_components=2, perplexity=40, random_state=42, n_iter=500)
    proj = tsne.fit_transform(latents_norm)

    ax = axes[task_id]
    for k in range(K):
        mask = labels_sub == k
        ax.scatter(proj[mask, 0], proj[mask, 1],
                   c=[colors[k]], s=8, alpha=0.6, label=f"WP {k}")

    ax.set_title(f"Task {task_id}: {SHORT_NAMES[task_id]}", fontsize=8, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if task_id == 0:
        ax.legend(fontsize=6, loc="upper right", markerscale=2)

fig.suptitle(f"t-SNE of Latent Space — Waypoint Assignments (k={K})",
    fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
tsne_path = os.path.join(OUT_DIR, "tsne_waypoints.png")
plt.savefig(tsne_path, dpi=150, bbox_inches="tight")
print(f"Saved t-SNE plot to {tsne_path}")


# ── 2. Waypoint timeline: show progression through a single demo ──────────────
# We need to know where each demo starts/ends in the latent array
# Dataset stores seq_len windows, so we reconstruct per-demo timeline
# by tracking when labels shift over the course of the trajectory

# For this we reload task 0 and show a few individual demo timelines
fig2, axes2 = plt.subplots(2, 5, figsize=(20, 6))
axes2 = axes2.flatten()

DEMO_LEN_APPROX = 100  # typical demo length in frames

for task_id in range(10):
    latents = np.load(os.path.join(LATENT_DIR, f"task{task_id}_latents.npy"))
    results = torch.load(os.path.join(LATENT_DIR, f"task{task_id}_waypoints.pt"))
    labels  = results[K]["labels"]

    ax = axes2[task_id]

    # show waypoint assignment over time for first ~5 demos
    # approximate: each demo contributes ~seq_len windows at end
    # just plot labels[:500] as a timeline
    n_show = min(500, len(labels))
    timeline = labels[:n_show]

    # color each timestep by waypoint
    for t, wp in enumerate(timeline):
        ax.axvline(t, color=colors[wp], alpha=0.4, linewidth=1.5)

    ax.set_xlim(0, n_show)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Frame", fontsize=7)
    ax.set_title(f"Task {task_id}: {SHORT_NAMES[task_id]}", fontsize=8, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # add waypoint color legend on first plot
    if task_id == 0:
        for k in range(K):
            ax.plot([], [], color=colors[k], linewidth=3, label=f"WP {k}")
        ax.legend(fontsize=6, loc="upper right")

fig2.suptitle(f"Waypoint Assignment Timeline — First 500 Frames per Task (k={K})",
    fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
timeline_path = os.path.join(OUT_DIR, "waypoint_timeline.png")
plt.savefig(timeline_path, dpi=150, bbox_inches="tight")
print(f"Saved timeline plot to {timeline_path}")

print("\nDone!")
