import os, numpy as np, matplotlib.pyplot as plt, matplotlib.image as mpimg
import subprocess, glob

video_dir = "/234/outputs/bc_baseline_full2/eval/videos_step_080000"

TASK_NAMES = [
    "Pick up black bowl\nbetween plate & ramekin",
    "Pick up black bowl\nnext to ramekin",
    "Pick up black bowl\nfrom table center",
    "Pick up black bowl\non cookie box",
    "Pick up black bowl\nin top drawer",
    "Pick up black bowl\non ramekin",
    "Pick up black bowl\nnext to cookie box",
    "Pick up black bowl\non stove",
    "Pick up black bowl\nnext to plate",
    "Pick up black bowl\non wooden cabinet",
]

frames = []
for i in range(10):
    pattern = os.path.join(video_dir, f"libero_spatial_{i}", "eval_episode_0.mp4")
    matches = glob.glob(pattern)
    if not matches:
        frames.append(None)
        continue
    video_path = matches[0]
    frame_path = f"/tmp/task_{i}_frame.png"
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-vf", "select=eq(n\\,0)",
        "-frames:v", "1", frame_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    frames.append(frame_path if os.path.exists(frame_path) else None)

fig, axes = plt.subplots(2, 5, figsize=(18, 7))
axes = axes.flatten()

for i, (frame_path, name) in enumerate(zip(frames, TASK_NAMES)):
    ax = axes[i]
    if frame_path is None:
        ax.text(0.5, 0.5, "No video", ha="center", va="center", transform=ax.transAxes)
    else:
        img = mpimg.imread(frame_path)
        ax.imshow(img)
    ax.set_title(f"Task {i}\n{name}", fontsize=8, fontweight="bold")
    ax.axis("off")

fig.suptitle("LIBERO-Spatial: Task Setup Overview",
    fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
out = "/234/outputs/bc_baseline_full2/task_setup_overview.png"
plt.savefig(out, bbox_inches="tight", dpi=150)
print(f"Saved to {out}")
