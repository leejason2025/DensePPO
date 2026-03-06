import os, sys, numpy as np, torch
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader

sys.path.insert(0, "/234/LIBERO")

from libero.libero import benchmark
from libero.lifelong.models.bc_transformer_policy import BCTransformerPolicy
from libero.lifelong.datasets import get_dataset, SequenceVLDataset
from libero.lifelong.utils import get_task_embs

EXP_DIR  = "/234/LIBERO/experiments/LIBERO_SPATIAL/Sequential/BCTransformerPolicy_seed42/run_003"
DATA_DIR = "/234/data/datasets/libero_spatial"
OUT_DIR  = "/234/outputs/waypoints"
K_VALUES = [3, 5, 10]
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

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

os.makedirs(OUT_DIR, exist_ok=True)

benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict["libero_spatial"]()


def load_policy_and_data(task_id):
    ckpt = torch.load(os.path.join(EXP_DIR, f"task{task_id}_model.pth"), map_location=DEVICE)
    cfg  = ckpt["cfg"]

    hdf5_path = os.path.join(DATA_DIR, f"{TASK_NAMES[task_id]}_demo.hdf5")
    dataset, shape_meta = get_dataset(
        dataset_path=hdf5_path,
        obs_modality=cfg.data.obs.modality,
        initialize_obs_utils=(task_id == 0),
        seq_len=cfg.data.seq_len,
    )

    task = task_suite.get_task(task_id)
    task_embs = get_task_embs(cfg, [task.language])
    task_emb  = task_embs[0].to(DEVICE)

    vl_dataset = SequenceVLDataset(dataset, task_emb)

    policy = BCTransformerPolicy(cfg, shape_meta)
    policy.load_state_dict(ckpt["state_dict"])
    policy.to(DEVICE)
    policy.eval()

    return policy, vl_dataset, task_emb, cfg


def extract_latents(policy, vl_dataset, task_emb):
    loader = DataLoader(vl_dataset, batch_size=64, shuffle=False, num_workers=0)
    all_latents = []

    with torch.no_grad():
        for batch in loader:
            data = {
                "obs":      {k: v.to(DEVICE) for k, v in batch["obs"].items()},
                "task_emb": batch["task_emb"].to(DEVICE),
            }
            x = policy.spatial_encode(data)   # (B, T, num_modalities, E)
            x = policy.temporal_encode(x)     # (B, T, E)
            z = x[:, -1].cpu().numpy()        # (B, E) — last timestep
            all_latents.append(z)

    return np.concatenate(all_latents, axis=0)


def cluster_waypoints(latents, k_values):
    latents_norm = normalize(latents)
    results = {}
    for k in k_values:
        print(f"    k={k}...", end=" ", flush=True)
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(latents_norm)
        results[k] = {
            "centers": km.cluster_centers_,
            "labels":  km.labels_,
            "inertia": km.inertia_,
        }
        print(f"inertia={km.inertia_:.2f}")
    return results


all_inertias = {k: [] for k in K_VALUES}

for task_id in range(10):
    print(f"\n=== Task {task_id}: {TASK_NAMES[task_id][:55]}... ===")
    policy, vl_dataset, task_emb, cfg = load_policy_and_data(task_id)
    print(f"  Dataset size: {len(vl_dataset)} samples")

    print("  Extracting latents...")
    latents = extract_latents(policy, vl_dataset, task_emb)
    print(f"  Latent shape: {latents.shape}")

    print("  Clustering...")
    results = cluster_waypoints(latents, K_VALUES)

    for k in K_VALUES:
        all_inertias[k].append(results[k]["inertia"])

    np.save(os.path.join(OUT_DIR, f"task{task_id}_latents.npy"), latents)
    torch.save(results, os.path.join(OUT_DIR, f"task{task_id}_waypoints.pt"))
    print(f"  Saved to {OUT_DIR}/task{task_id}_waypoints.pt")

# elbow plot
fig, ax = plt.subplots(figsize=(8, 5))
for k in K_VALUES:
    ax.plot(range(10), all_inertias[k], marker="o", linewidth=2, label=f"k={k}")
ax.set_xlabel("Task ID", fontsize=12)
ax.set_ylabel("K-Means Inertia", fontsize=12)
ax.set_title("Clustering Inertia per Task\n(lower = tighter clusters)", fontsize=13, fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "clustering_inertia.png"), dpi=150, bbox_inches="tight")
print(f"\nElbow plot saved. Done!")
