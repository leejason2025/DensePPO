"""
3D PCA visualization of BC latent space for Task 6.
- L2-normalized latents to match training
- Renders a task setup frame from the environment
- Side by side: task image + 3D PCA plot
"""
import os, sys, numpy as np, torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import h5py

sys.path.insert(0, "/234/LIBERO")

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.lifelong.models.bc_transformer_policy import BCTransformerPolicy
from libero.lifelong.datasets import get_dataset
from libero.lifelong.utils import get_task_embs

TASK_ID  = 6
EXP_DIR  = "/234/LIBERO/experiments/LIBERO_SPATIAL/Sequential/BCTransformerPolicy_seed42/run_003"
WP_DIR   = "/234/outputs/waypoints"
DATA_DIR = "/234/data/datasets/libero_spatial"
BDDL_DIR = "/234/LIBERO/libero/libero/bddl_files/libero_spatial"
OUT_DIR  = "/234/outputs/plots"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN  = 10
K        = 5

os.makedirs(OUT_DIR, exist_ok=True)

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

benchmark_dict = benchmark.get_benchmark_dict()
task_suite     = benchmark_dict["libero_spatial"]()

# ── Render task setup frame ───────────────────────────────────────────────────
print("Rendering task setup frame...")
task = task_suite.get_task(TASK_ID)
env  = OffScreenRenderEnv(
    bddl_file_name=os.path.join(BDDL_DIR, task.bddl_file),
    camera_heights=256,
    camera_widths=256,
)
obs        = env.reset()
task_frame = obs["agentview_image"]
env.close()
print("Task frame captured.")

# ── Load BC policy ────────────────────────────────────────────────────────────
print("Loading BC policy...")
ckpt = torch.load(os.path.join(EXP_DIR, f"task{TASK_ID}_model.pth"), map_location=DEVICE)
cfg  = ckpt["cfg"]
_, shape_meta = get_dataset(
    dataset_path=os.path.join(DATA_DIR, f"{TASK_NAMES[TASK_ID]}_demo.hdf5"),
    obs_modality=cfg.data.obs.modality,
    initialize_obs_utils=True,
    seq_len=cfg.data.seq_len,
)
task_emb = get_task_embs(cfg, [task.language])[0].to(DEVICE)
policy   = BCTransformerPolicy(cfg, shape_meta)
policy.load_state_dict(ckpt["state_dict"])
policy.to(DEVICE)
policy.eval()
for p in policy.parameters():
    p.requires_grad = False

# ── Extract latents per demo ──────────────────────────────────────────────────
print("Extracting demo latents...")
hdf5_path        = os.path.join(DATA_DIR, f"{TASK_NAMES[TASK_ID]}_demo.hdf5")
all_demo_latents = []

with h5py.File(hdf5_path, "r") as f:
    demo_keys = sorted(f["data"].keys())
    for demo_key in demo_keys:
        demo = f["data"][demo_key]
        T    = demo["actions"].shape[0]

        agentview   = torch.tensor(np.array(demo["obs"]["agentview_rgb"]),
                                   dtype=torch.float32).permute(0,3,1,2) / 255.0
        eye_in_hand = torch.tensor(np.array(demo["obs"]["eye_in_hand_rgb"]),
                                   dtype=torch.float32).permute(0,3,1,2) / 255.0
        joint_pos   = torch.tensor(np.array(demo["obs"]["joint_states"]),   dtype=torch.float32)
        gripper     = torch.tensor(np.array(demo["obs"]["gripper_states"]), dtype=torch.float32)

        av_buf = agentview[0].unsqueeze(0).repeat(SEQ_LEN,1,1,1)
        eh_buf = eye_in_hand[0].unsqueeze(0).repeat(SEQ_LEN,1,1,1)
        jp_buf = joint_pos[0].unsqueeze(0).repeat(SEQ_LEN,1)
        gp_buf = gripper[0].unsqueeze(0).repeat(SEQ_LEN,1)

        demo_latents = []
        for t in range(T):
            if t > 0:
                av_buf = torch.cat([av_buf[1:], agentview[t].unsqueeze(0)],   dim=0)
                eh_buf = torch.cat([eh_buf[1:], eye_in_hand[t].unsqueeze(0)], dim=0)
                jp_buf = torch.cat([jp_buf[1:], joint_pos[t].unsqueeze(0)],   dim=0)
                gp_buf = torch.cat([gp_buf[1:], gripper[t].unsqueeze(0)],     dim=0)

            data_dict = {
                "obs": {
                    "agentview_rgb":   av_buf.unsqueeze(0).to(DEVICE),
                    "eye_in_hand_rgb": eh_buf.unsqueeze(0).to(DEVICE),
                    "joint_states":    jp_buf.unsqueeze(0).to(DEVICE),
                    "gripper_states":  gp_buf.unsqueeze(0).to(DEVICE),
                },
                "task_emb": task_emb.unsqueeze(0).to(DEVICE),
            }
            with torch.no_grad():
                x = policy.spatial_encode(data_dict)
                x = policy.temporal_encode(x)
                z = x[0, -1].cpu().numpy()
            demo_latents.append(z)

        all_demo_latents.append(np.array(demo_latents))
        print(f"  {demo_key}: {T} steps", end="\r")

print(f"\nExtracted {len(all_demo_latents)} demos.")

# ── Load waypoints ────────────────────────────────────────────────────────────
wp_results     = torch.load(os.path.join(WP_DIR, f"task{TASK_ID}_waypoints.pt"))
waypoints_raw  = wp_results[K]["centers"]  # (K, 64)

# ── L2 normalize everything to match training ─────────────────────────────────
print("Normalizing latents...")
all_demo_latents_norm = [normalize(traj, norm="l2") for traj in all_demo_latents]
all_latents_norm      = np.vstack(all_demo_latents_norm)
waypoints_norm        = normalize(waypoints_raw, norm="l2")

# ── PCA ───────────────────────────────────────────────────────────────────────
print("Fitting PCA...")
pca           = PCA(n_components=3)
pca.fit(all_latents_norm)
var_explained = pca.explained_variance_ratio_
print(f"Variance explained: PC1={var_explained[0]:.1%}, PC2={var_explained[1]:.1%}, PC3={var_explained[2]:.1%}")

demo_pca     = [pca.transform(traj) for traj in all_demo_latents_norm]
waypoint_pca = pca.transform(waypoints_norm)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 7))

# left panel: task image
ax_img = fig.add_subplot(1, 2, 1)
ax_img.imshow(task_frame)
ax_img.axis("off")
ax_img.set_title("Task 6: Pick Up Black Bowl\nNext to Cookie Box → Place on Plate",
                 fontsize=12, fontweight="bold", pad=12)

# right panel: 3D PCA
ax3d   = fig.add_subplot(1, 2, 2, projection="3d")
cmap   = plt.cm.cool
n_demos = len(demo_pca)

for i, traj in enumerate(demo_pca):
    color = cmap(i / n_demos)
    ax3d.plot(traj[:, 0], traj[:, 1], traj[:, 2],
              color=color, alpha=0.2, linewidth=0.7)
    ax3d.scatter(traj[0, 0],  traj[0, 1],  traj[0, 2],
                 color=color, s=8,  alpha=0.3, zorder=2)
    ax3d.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2],
                 color=color, s=12, alpha=0.5, marker="^", zorder=2)

wp_colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#3498db"]
for i, (wp, color) in enumerate(zip(waypoint_pca, wp_colors)):
    ax3d.scatter(wp[0], wp[1], wp[2],
                 color=color, s=300, marker="*",
                 edgecolors="black", linewidth=0.8,
                 zorder=5, label=f"WP{i+1}")
    ax3d.text(wp[0], wp[1], wp[2] + 0.015,
              f"WP{i+1}", fontsize=8, fontweight="bold",
              color=color, ha="center", zorder=6)

for i in range(K - 1):
    ax3d.plot([waypoint_pca[i,0], waypoint_pca[i+1,0]],
              [waypoint_pca[i,1], waypoint_pca[i+1,1]],
              [waypoint_pca[i,2], waypoint_pca[i+1,2]],
              "k--", linewidth=1.0, alpha=0.5, zorder=4)

ax3d.set_xlabel(f"PC1 ({var_explained[0]:.1%})", fontsize=9, labelpad=6)
ax3d.set_ylabel(f"PC2 ({var_explained[1]:.1%})", fontsize=9, labelpad=6)
ax3d.set_zlabel(f"PC3 ({var_explained[2]:.1%})", fontsize=9, labelpad=6)
ax3d.set_title(f"BC Latent Space (L2-normalized) — 3D PCA\n"
               f"50 demo trajectories + K-means waypoints (k={K})\n"
               f"Silhouette = 0.292  |  Var. explained: {sum(var_explained):.1%}",
               fontsize=10, fontweight="bold")
ax3d.legend(fontsize=8, loc="upper left", title="Waypoints", title_fontsize=8)
ax3d.view_init(elev=25, azim=45)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/latent_pca_3d_task6.png", dpi=150, bbox_inches="tight")
print("Saved latent_pca_3d_task6.png")
plt.close()
print("Done.")
