# CS 234 RL Project: Unsupervised Dense Reward Generation
# Full Project Log — LIBERO-Spatial Benchmark

---

## PHASE 1: BC BASELINE

### Goal
Train a Behavioral Cloning baseline on LIBERO-Spatial (10 tasks, 50 demos each)
targeting 70-80% success rate to match published baselines.

---

### Step 1: Initial Attempt with LeRobot ACT
- Installed LeRobot in Python 3.13 venv at `/234/venv`
- Downloaded LIBERO-Spatial dataset via HuggingFace: `HuggingFaceVLA/libero`
- Ran ACT policy training with default settings

**Issues faced:**
- First run used only 10 episodes (843 frames) by accident — appeared to work
  (loss 5.556 → 0.093, eval 39.5%) but was just overfitting a tiny subset
- Learning rate stuck at 1e-5 (ACTConfig default), could not override via CLI flags
  - Tried `--policy.optimizer_lr=1e-4` — did not work
  - Tried `--optimizer.lr=1e-4` — did not work
  - Root cause: LeRobot CLI flag parsing does not override nested config defaults
- Full 500-demo run (120k steps) converged to only 31.5% success
- Run crashed at end with HuggingFace 401 auth error (checkpoints already saved)
- **Conclusion:** LeRobot ACT cannot easily reach 80% on LIBERO-Spatial without
  patching the source to fix LR scheduler

**Results:**
- LeRobot ACT (120k steps, 500 demos): 31.5% on LIBERO-Spatial
- Checkpoints: `/234/outputs/bc_baseline_full2/checkpoints/120000/`

---

### Step 2: Switch to Official LIBERO Codebase

**Why:** Official LIBERO ResNet-Transformer uses cosine LR schedule (1e-4 → 1e-5)
which is critical for convergence. LeRobot does not expose this easily.

**Installation steps:**
```bash
# 1. Install Miniconda (server only had Python 3.13, LIBERO needs 3.8)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /234/miniconda.sh
bash /234/miniconda.sh -b -p /234/miniconda

# 2. Accept Anaconda ToS (new requirement)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# 3. Create Python 3.8 environment
conda create -n libero python=3.8.13 -y
source /234/miniconda/bin/activate libero

# 4. Install dependencies
cd /234/LIBERO
pip install -r requirements.txt

# 5. Upgrade PyTorch (default 1.11 too old for server kernel)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# 6. Install LIBERO as package
pip install -e .

# 7. Download LIBERO-Spatial dataset
python benchmark_scripts/download_libero_datasets.py \
  --datasets libero_spatial \
  --download-dir /234/data/datasets
```

**Bugs fixed:**
- `/234/LIBERO/libero/lifelong/algos/base.py` line 145:
  Changed `persistent_workers=True` → `persistent_workers=False`
  (conflicts with num_workers=0)
- Disabled multiprocessing for eval (h5py pickle errors):
  Added `eval.num_workers=0 eval.use_mp=false` to training command

**Training command:**
```bash
source /234/miniconda/bin/activate libero
cd /234/LIBERO
python libero/lifelong/main.py \
  seed=42 \
  benchmark_name=LIBERO_SPATIAL \
  policy=bc_transformer_policy \
  lifelong=base \
  train.num_workers=0 \
  eval.num_workers=0 \
  eval.use_mp=false
```

**Hyperparameters:**
- Policy: BCTransformerPolicy (ResNet-18 + Transformer)
- Optimizer: AdamW
- Learning rate: 1e-4 cosine decay to 1e-5
- Batch size: 32
- Epochs: 50 per task
- Eval every: 5 epochs, 20 rollouts per task
- Demos: 50 per task (500 total)
- Image size: 128x128
- Sequence length: 10 frames
- Action head: GMM with 5 modes
- Seed: 42

**Results (run_003):**
| Task | Description              | Best Success |
|------|--------------------------|-------------|
| 0    | Between plate & ramekin  | 80%         |
| 1    | Next to ramekin          | 80%         |
| 2    | Table center             | 100%        |
| 3    | On cookie box            | 90%         |
| 4    | Top drawer               | 75%         |
| 5    | On ramekin               | 70%         |
| 6    | Next to cookie box       | 85%         |
| 7    | On stove                 | 90%         |
| 8    | Next to plate            | 80%         |
| 9    | On wooden cabinet        | 65%         |
| AVG  |                          | **81.5%**   |

- Checkpoints: `/234/LIBERO/experiments/LIBERO_SPATIAL/Sequential/BCTransformerPolicy_seed42/run_003/`
- Per-task models: `task{0-9}_model.pth`
- Training logs: `task{0-9}_auc.log`
- Results: `result.pt`

---

## PHASE 2: LATENT WAYPOINT EXTRACTION

### Goal
Extract meaningful intermediate states (waypoints) from demo trajectories
using the frozen BC encoder, to use as dense reward signal for PPO.

### Why
LIBERO only provides sparse reward (+1 on task completion).
Sparse reward makes PPO extremely slow/impossible to learn from.
Waypoints extracted from demonstrations give dense reward without human labeling.

### Steps

**1. Inspect checkpoint format**
```bash
python3 -c "
import torch
ckpt = torch.load('.../task0_model.pth')
print(ckpt.keys())  # state_dict, cfg, previous_masks
"
```

**2. Understand model architecture**
- `encoders` — ResNet-18 image encoders (agentview + eye-in-hand)
- `language_encoder` — BERT language encoder
- `extra_encoder` — proprioception (joints + gripper)
- `temporal_transformer` — transformer over time
- `policy_head` — GMM action head (not used for extraction)

**Issues faced:**
- `BCTransformerPolicy(cfg, None)` crashes — needs `shape_meta`
  Fix: use `get_dataset()` to obtain `shape_meta` properly
- `get_task_embs(cfg, [task])` crashes — expects list of strings not task objects
  Fix: pass `[task.language]` instead
- `policy.extra_encoder(proprio)` crashes with dimension error
  Fix: inspected `spatial_encode()` and `temporal_encode()` source —
  must pass full `data` dict with `obs` and `task_emb` keys,
  not call sub-modules manually
- sklearn not installed: `pip install scikit-learn`

**Extraction method:**
- Load each task model + dataset via LIBERO's `get_dataset()`
- Run batches through `policy.spatial_encode()` + `policy.temporal_encode()`
- Take last timestep of transformer output as state representation
- Latent dimension: 64

**Clustering:**
- L2 normalize latents before clustering
- K-means with k=3, 5, 10 (ablation)
- Results saved per task

**Script:** `/234/scripts/extract_waypoints.py`

---

## PHASE 3: REWARD QUALITY EVALUATION

### Goal
Verify that waypoints capture meaningful task structure before using them as reward.

### Steps

**1. t-SNE visualization**
- Subsample 2000 points per task
- Project 64-dim latents to 2D
- Color by cluster assignment
- Result: clearly separated blobs — good structure confirmed

**2. Temporal ordering fix**
- Initial timeline showed random color flickering — waypoints had no temporal order
- Root cause: k-means assigns cluster IDs arbitrarily
- Fix: re-order waypoints by mean temporal position in demos
- After fix: smooth color transitions early→late confirmed

**3. Silhouette scores (cluster quality metric, -1 to +1)**
| Task | Score  |
|------|--------|
| 0    | 0.236  |
| 1    | 0.190  |
| 2    | 0.269  |
| 3    | 0.251  |
| 4    | 0.167  |
| 5    | 0.232  |
| 6    | 0.292  |
| 7    | 0.254  |
| 8    | 0.179  |
| 9    | 0.200  |
| MEAN | **0.227** |

- All tasks in acceptable-to-good range
- Task 4 (top drawer) lowest — most complex task, messiest latent space
- Task 6 (next to cookie box) highest — cleanest pick-and-place trajectory

**Scripts:** `/234/scripts/visualize_waypoints.py`, `/234/scripts/evaluate_waypoints.py`

---

## PHASE 4: RESIDUAL PPO (TODO)

### Plan
- Start with Task 2 (100% BC) and Task 4 (75% BC) to validate
- Freeze BC policy as base
- Train small residual network on top for action corrections
- Dense reward = progress toward next unvisited waypoint (k=5)
- Bonus reward on waypoint reached + sparse +1 on task completion
- Library: Stable-Baselines3

---

## FILE LOCATIONS SUMMARY

| File | Path |
|------|------|
| BC models | `/234/LIBERO/experiments/.../run_003/task{0-9}_model.pth` |
| Training logs | `/234/LIBERO/experiments/.../run_003/task{0-9}_auc.log` |
| LIBERO dataset | `/234/data/datasets/libero_spatial/` |
| Waypoint clusters | `/234/outputs/waypoints/task{0-9}_waypoints.pt` |
| Latent vectors | `/234/outputs/waypoints/task{0-9}_latents.npy` |
| All plots | `/234/outputs/waypoints/*.png` |
| All scripts | `/234/scripts/` |
| LeRobot venv | `/234/venv/` |
| Conda env | `/234/miniconda/envs/libero/` |

## KEY COMMANDS
```bash
# Activate LIBERO env
source /234/miniconda/bin/activate libero

# Activate LeRobot env
source /234/venv/bin/activate

# Run BC training (from /234/LIBERO)
python libero/lifelong/main.py seed=42 benchmark_name=LIBERO_SPATIAL \
  policy=bc_transformer_policy lifelong=base \
  train.num_workers=0 eval.num_workers=0 eval.use_mp=false

# Re-run waypoint extraction
python3 /234/scripts/extract_waypoints.py

# Re-run reward evaluation
python3 /234/scripts/evaluate_waypoints.py
```
