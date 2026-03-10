"""
Training curves: episode reward vs timesteps for Dense PPO and Sparse PPO.
Each task = thin line, bold line = average across all 10 tasks.
"""
import re, os, numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

LOG_DIR = "/234/outputs/ppo_logs"
OUT_DIR = "/234/outputs/plots"
os.makedirs(OUT_DIR, exist_ok=True)

DENSE_COLOR  = "#2980b9"
SPARSE_COLOR = "#e67e22"


def parse_log(logfile):
    data = []
    with open(logfile) as f:
        for line in f:
            m = re.search(r'num_timesteps=(\d+), episode_reward=([-\d.]+)', line)
            if m:
                data.append((int(m.group(1)), float(m.group(2))))
    return data


def interpolate_to_grid(data, grid):
    if not data:
        return np.full(len(grid), np.nan)
    ts  = np.array([d[0] for d in data])
    rew = np.array([d[1] for d in data])
    return np.interp(grid, ts, rew, left=np.nan, right=np.nan)


# ── Load all logs ─────────────────────────────────────────────────────────────
dense_data  = {}
sparse_data = {}

for task_id in range(10):
    dense_log  = os.path.join(LOG_DIR, f"task{task_id}.log")
    sparse_log = os.path.join(LOG_DIR, f"bc_vanilla_task{task_id}.log")
    if os.path.exists(dense_log):
        d = parse_log(dense_log)
        if d: dense_data[task_id] = d
    if os.path.exists(sparse_log):
        d = parse_log(sparse_log)
        if d: sparse_data[task_id] = d

print(f"Dense logs loaded:  {sorted(dense_data.keys())}")
print(f"Sparse logs loaded: {sorted(sparse_data.keys())}")

# ── Build common grids ────────────────────────────────────────────────────────
dense_max  = max(d[-1][0] for d in dense_data.values())
sparse_max = max(d[-1][0] for d in sparse_data.values())

dense_grid  = np.linspace(10000, dense_max,  100)
sparse_grid = np.linspace(10000, sparse_max, 100)

dense_interp  = {tid: interpolate_to_grid(d, dense_grid)  for tid, d in dense_data.items()}
sparse_interp = {tid: interpolate_to_grid(d, sparse_grid) for tid, d in sparse_data.items()}

dense_avg  = np.nanmean(np.array(list(dense_interp.values())),  axis=0)
sparse_avg = np.nanmean(np.array(list(sparse_interp.values())), axis=0)

dense_avg_smooth  = uniform_filter1d(dense_avg,  size=5)
sparse_avg_smooth = uniform_filter1d(sparse_avg, size=5)

# shared y range
all_vals = [v for arr in list(dense_interp.values()) + list(sparse_interp.values())
            for v in arr if not np.isnan(v)]
ymin = min(all_vals)
ymax = max(all_vals)
pad  = (ymax - ymin) * 0.05


# ── Plot 1: Both on same axes ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))

for arr in dense_interp.values():
    mask = ~np.isnan(arr)
    ax.plot(dense_grid[mask] / 1000, arr[mask], color=DENSE_COLOR, alpha=0.2, linewidth=0.9)

for arr in sparse_interp.values():
    mask = ~np.isnan(arr)
    ax.plot(sparse_grid[mask] / 1000, arr[mask], color=SPARSE_COLOR, alpha=0.2, linewidth=0.9)

ax.plot(dense_grid / 1000,  dense_avg_smooth,  color=DENSE_COLOR,  linewidth=3.0, label="BC + Dense PPO (avg)", zorder=5)
ax.plot(sparse_grid / 1000, sparse_avg_smooth, color=SPARSE_COLOR, linewidth=3.0, label="BC + Sparse PPO (avg)", zorder=5)
ax.plot([], [], color=DENSE_COLOR,  alpha=0.4, linewidth=1.5, label="BC + Dense PPO (per task)")
ax.plot([], [], color=SPARSE_COLOR, alpha=0.4, linewidth=1.5, label="BC + Sparse PPO (per task)")

ax.set_xlabel("Timesteps (thousands)", fontsize=12)
ax.set_ylabel("Episode Reward", fontsize=12)
ax.set_title("Training Curves: Episode Reward vs Timesteps\nBold = average across 10 tasks, thin = individual tasks",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10, loc="lower right")
ax.grid(alpha=0.3)
ax.set_xlim(10, 500)
ax.set_ylim(ymin - pad, ymax + pad)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/training_curves_reward.png", dpi=150, bbox_inches="tight")
print("Saved training_curves_reward.png")
plt.close()


# ── Plot 2: Side by side, shared y axis, capped at 180k ──────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for arr in dense_interp.values():
    mask = ~np.isnan(arr)
    ax1.plot(dense_grid[mask] / 1000, arr[mask], color=DENSE_COLOR, alpha=0.25, linewidth=1.0)
ax1.plot(dense_grid / 1000, dense_avg_smooth, color=DENSE_COLOR, linewidth=3.5, zorder=5)
ax1.set_xlabel("Timesteps (thousands)", fontsize=11)
ax1.set_ylabel("Episode Reward", fontsize=11)
ax1.set_title("BC + Dense PPO\n(waypoint reward)", fontsize=12, fontweight="bold", color=DENSE_COLOR)
ax1.legend(["Average", "Per task"], fontsize=9, loc="lower right")
ax1.grid(alpha=0.3)
ax1.set_xlim(10, 500)
ax1.set_ylim(ymin - pad, ymax + pad)
dense_at_end = dense_avg_smooth[np.argmin(np.abs(dense_grid - 500000))]
ax1.annotate(f"Avg @ 500k: {dense_at_end:.2f}",
             xy=(500, dense_at_end), xytext=(-100, 15), textcoords="offset points",
             fontsize=10, color=DENSE_COLOR, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=DENSE_COLOR))

for arr in sparse_interp.values():
    mask = ~np.isnan(arr)
    ax2.plot(sparse_grid[mask] / 1000, arr[mask], color=SPARSE_COLOR, alpha=0.25, linewidth=1.0)
ax2.plot(sparse_grid / 1000, sparse_avg_smooth, color=SPARSE_COLOR, linewidth=3.5, zorder=5)
ax2.set_xlabel("Timesteps (thousands)", fontsize=11)
ax2.set_ylabel("Episode Reward", fontsize=11)
ax2.set_title("BC + Sparse PPO\n(sparse reward only)", fontsize=12, fontweight="bold", color=SPARSE_COLOR)
ax2.legend(["Average", "Per task"], fontsize=9, loc="lower right")
ax2.grid(alpha=0.3)
ax2.set_xlim(10, 180)
ax2.set_ylim(ymin - pad, ymax + pad)
sparse_at_end = sparse_avg_smooth[np.argmin(np.abs(sparse_grid - 180000))]
ax2.annotate(f"Avg @ 180k: {sparse_at_end:.2f}",
             xy=(180, sparse_at_end), xytext=(-100, 15), textcoords="offset points",
             fontsize=10, color=SPARSE_COLOR, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=SPARSE_COLOR))

fig.suptitle("Training Curves: BC + Dense PPO vs BC + Sparse PPO\nBold = average across 10 tasks, thin = individual tasks",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/training_curves_sidebyside.png", dpi=150, bbox_inches="tight")
print("Saved training_curves_sidebyside.png")
plt.close()


# ── Plot 3: Per-task grid ─────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(18, 7))
axes = axes.flatten()

for task_id in range(10):
    ax = axes[task_id]
    if task_id in dense_interp:
        arr  = dense_interp[task_id]
        mask = ~np.isnan(arr)
        ax.plot(dense_grid[mask] / 1000, uniform_filter1d(arr, size=3)[mask],
                color=DENSE_COLOR, linewidth=2.5, label="Dense PPO")
    if task_id in sparse_interp:
        arr  = sparse_interp[task_id]
        mask = ~np.isnan(arr)
        ax.plot(sparse_grid[mask] / 1000, uniform_filter1d(arr, size=3)[mask],
                color=SPARSE_COLOR, linewidth=2.5, label="Sparse PPO")
    ax.set_title(f"Task {task_id}", fontsize=10, fontweight="bold")
    ax.set_xlabel("Timesteps (k)", fontsize=8)
    ax.set_ylabel("Episode Reward", fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(10, 500)
    ax.set_ylim(ymin - pad, ymax + pad)
    if task_id == 0:
        ax.legend(fontsize=8)

fig.suptitle("Per-Task Training Curves: BC + Dense PPO vs BC + Sparse PPO",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/training_curves_pertask.png", dpi=150, bbox_inches="tight")
print("Saved training_curves_pertask.png")
plt.close()

print("\nAll training curve plots saved.")
