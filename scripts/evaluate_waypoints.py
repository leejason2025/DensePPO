"""
Waypoint Quality Evaluation
- Re-orders waypoints by mean temporal position
- Shows clean timeline with ordered waypoints
- Shows average waypoint visit time with std bars
- Shows silhouette score (cluster quality metric)
"""
import os, sys, numpy as np, torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score

OUT_DIR    = "/234/outputs/waypoints"
K          = 5

SHORT_NAMES = [
    "Between plate & ramekin", "Next to ramekin", "Table center",
    "On cookie box", "Top drawer", "On ramekin",
    "Next to cookie box", "On stove", "Next to plate", "On wooden cabinet",
]

# approximate demo lengths from dataset sizes and seq_len=10
# each frame t in a demo of length L contributes 1 sample
# so we can reconstruct per-demo frame indices
import h5py
DATA_DIR   = "/234/data/datasets/libero_spatial"
TASK_NAMES = [
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
]

def get_demo_lengths(task_id):
    """Return list of demo lengths from HDF5."""
    path = os.path.join(DATA_DIR, f"{TASK_NAMES[task_id]}_demo.hdf5")
    with h5py.File(path, "r") as f:
        return [f["data"][k]["actions"].shape[0] for k in sorted(f["data"].keys())]

def reorder_waypoints_by_time(labels, demo_lengths):
    """
    Assign a mean temporal position (0-1 normalized) to each cluster,
    then return a mapping old_label -> new_label sorted by time.
    """
    # build per-frame normalized time index
    frame_times = []
    for L in demo_lengths:
        frame_times.extend(np.linspace(0, 1, L).tolist())
    frame_times = np.array(frame_times[:len(labels)])

    # mean time for each cluster
    mean_times = []
    for k in range(K):
        mask = labels == k
        mean_times.append(frame_times[mask].mean() if mask.sum() > 0 else 0)

    # sort clusters by mean time
    order = np.argsort(mean_times)
    remap = {old: new for new, old in enumerate(order)}
    new_labels = np.array([remap[l] for l in labels])
    return new_labels, np.array(mean_times)[order], frame_times


# ── Compute everything ────────────────────────────────────────────────────────
all_sil    = []
all_mean_times = []  # (10, K) — mean visit time per waypoint per task

for task_id in range(10):
    latents = np.load(os.path.join(OUT_DIR, f"task{task_id}_latents.npy"))
    results = torch.load(os.path.join(OUT_DIR, f"task{task_id}_waypoints.pt"))
    labels  = results[K]["labels"]
    demo_lengths = get_demo_lengths(task_id)

    new_labels, mean_times, frame_times = reorder_waypoints_by_time(labels, demo_lengths)

    # silhouette score: measures cluster separation (-1 bad, +1 perfect)
    latents_norm = normalize(latents)
    sub = min(2000, len(latents_norm))
    idx = np.random.choice(len(latents_norm), sub, replace=False)
    sil = silhouette_score(latents_norm[idx], new_labels[idx])
    all_sil.append(sil)
    all_mean_times.append(mean_times)

    # save reordered labels back
    results[K]["labels_ordered"] = new_labels
    torch.save(results, os.path.join(OUT_DIR, f"task{task_id}_waypoints.pt"))

all_mean_times = np.array(all_mean_times)  # (10, K)

print("Silhouette scores per task (higher=better, max=1.0):")
for i, s in enumerate(all_sil):
    print(f"  Task {i} ({SHORT_NAMES[i][:25]}): {s:.3f}")
print(f"  Mean: {np.mean(all_sil):.3f}")


# ── Figure 1: Clean timeline with ordered waypoints ───────────────────────────
colors = cm.RdYlGn(np.linspace(0.1, 0.9, K))  # red=early, green=late

fig, axes = plt.subplots(2, 5, figsize=(20, 5))
axes = axes.flatten()

for task_id in range(10):
    latents = np.load(os.path.join(OUT_DIR, f"task{task_id}_latents.npy"))
    results = torch.load(os.path.join(OUT_DIR, f"task{task_id}_waypoints.pt"))
    labels  = results[K]["labels_ordered"]
    demo_lengths = get_demo_lengths(task_id)

    # show first 3 demos as stacked horizontal bars
    ax = axes[task_id]
    ptr = 0
    for demo_idx, L in enumerate(demo_lengths[:5]):
        demo_labels = labels[ptr:ptr+L]
        for t, wp in enumerate(demo_labels):
            ax.barh(demo_idx, 1, left=t, color=colors[wp], height=0.8, linewidth=0)
        ptr += L

    ax.set_xlim(0, max(demo_lengths[:5]))
    ax.set_yticks(range(5))
    ax.set_yticklabels([f"Demo {i}" for i in range(5)], fontsize=6)
    ax.set_xlabel("Frame", fontsize=7)
    ax.set_title(f"Task {task_id}: {SHORT_NAMES[task_id]}", fontsize=8, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=colors[k], label=f"WP {k}") for k in range(K)]
fig.legend(handles=legend_elements, loc="lower center", ncol=K,
           fontsize=9, title="Waypoint (ordered early→late)", title_fontsize=9,
           bbox_to_anchor=(0.5, -0.05))

fig.suptitle(f"Waypoint Timeline — First 5 Demos per Task (k={K}, ordered by time)",
    fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "timeline_ordered.png"), dpi=150, bbox_inches="tight")
print("Saved ordered timeline.")


# ── Figure 2: Mean waypoint visit time per task ───────────────────────────────
fig2, ax2 = plt.subplots(figsize=(12, 5))

x = np.arange(10)
width = 0.15
for k in range(K):
    ax2.bar(x + k * width, all_mean_times[:, k], width,
            color=colors[k], label=f"WP {k}", alpha=0.85)

ax2.set_xticks(x + width * 2)
ax2.set_xticklabels([f"T{i}" for i in range(10)], fontsize=9)
ax2.set_ylabel("Mean Normalized Visit Time (0=start, 1=end)", fontsize=10)
ax2.set_title("When Each Waypoint Is Visited — All Tasks\n(good waypoints should spread evenly from 0→1)",
    fontsize=12, fontweight="bold")
ax2.legend(fontsize=8)
ax2.set_ylim(0, 1)
ax2.grid(True, alpha=0.3, axis="y")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "waypoint_visit_times.png"), dpi=150, bbox_inches="tight")
print("Saved visit times plot.")


# ── Figure 3: Silhouette scores ───────────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(10, 4))
bars = ax3.bar(range(10), all_sil,
               color=["green" if s > 0.2 else "orange" if s > 0.1 else "red" for s in all_sil],
               alpha=0.8)
ax3.axhline(np.mean(all_sil), color="black", linewidth=2,
            linestyle="--", label=f"Mean: {np.mean(all_sil):.3f}")
ax3.set_xticks(range(10))
ax3.set_xticklabels([f"T{i}\n{SHORT_NAMES[i][:12]}" for i in range(10)], fontsize=8)
ax3.set_ylabel("Silhouette Score", fontsize=11)
ax3.set_title("Cluster Quality per Task\n(>0.2 good, >0.1 acceptable, <0.1 poor)",
    fontsize=12, fontweight="bold")
ax3.set_ylim(-0.1, 0.6)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, axis="y")
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "silhouette_scores.png"), dpi=150, bbox_inches="tight")
print("Saved silhouette scores.")
print("\nAll done!")
