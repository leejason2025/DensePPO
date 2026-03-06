# CS234 Project: Setup & Run Documentation
## Unsupervised Dense Reward Generation for RL Fine-Tuning of BC Policies
**Repo:** https://github.com/leejason2025/DensePPO

---

## 1. Server & Hardware Info

| Component | Details |
|-----------|---------|
| GPU | NVIDIA GeForce RTX 3090 Ti |
| VRAM | 24 GB |
| RAM | 60 GB |
| CUDA Version | 12.8 |
| Driver Version | 570.195.03 |
| Python | 3.8.13 (conda env `libero`) |

---

## 2. Repo Structure

```
DensePPO/
├── .gitignore
├── requirements.txt
├── DOCUMENTATION.md
│
├── scripts/
│   ├── eval_ppo.py                  # Evaluate residual PPO models
│   ├── eval_vanilla_ppo.py          # Evaluate vanilla PPO ablation models
│   ├── evaluate_waypoints.py        # Compute silhouette scores, timeline plots
│   ├── residual_ppo.py              # Residual PPO (multi-task version)
│   ├── residual_ppo_single.py       # Residual PPO (single task, --task arg)
│   ├── vanilla_ppo_single.py        # Vanilla PPO baseline (--task arg)
│   └── visualization/
│       ├── extract_frames.py
│       ├── extract_waypoints.py     # Run BC encoder on demos, K-means cluster
│       ├── plot_combined.py         # BC training curves
│       ├── plot_loss.py
│       └── visualize_waypoints.py   # t-SNE + waypoint timeline plots
│
└── outputs/
    ├── eval_20260305_214058.txt      # Saved eval run (residual PPO)
    ├── eval_vanilla_20260306_122357.txt  # Saved eval run (vanilla PPO)
    ├── ppo_logs/
    │   ├── task0.log – task9.log    # Residual PPO training logs 
    │   └── vanilla_task0.log – vanilla_task9.log
    └── waypoints/
        └── task0_waypoints.pt – task9_waypoints.pt  # K-means waypoint centers
```

> Large files excluded from repo via `.gitignore`: `.pth`, `.zip`, `.npy`, `.hdf5`, datasets, checkpoints

---

## 3. Environment Setup

### Step 1: Install Miniconda
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p /234/miniconda
eval "$(/234/miniconda/bin/conda shell.bash hook)"
```

### Step 2: Accept Conda Terms of Service
```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

### Step 3: Create Python 3.8 Environment
```bash
conda create -n libero python=3.8.13 -y
source /234/miniconda/bin/activate libero
```

### Step 4: Install PyTorch (CUDA 11.8)
```bash
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```
> The default torch==1.11 in LIBERO's requirements.txt is too old for the server kernel. Must upgrade to 2.0.1.

### Step 5: Clone and Install LIBERO
```bash
cd /234
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -r requirements.txt
pip install -e .
```

### Step 6: Install Additional Dependencies
```bash
pip install scikit-learn
pip install stable-baselines3==2.3.2
pip install gymnasium==0.29.1 shimmy>=0.2.1
pip install tqdm rich
```

### Step 7: Apply Bug Fix to LIBERO Source
```bash
# Fix: persistent_workers=True crashes when num_workers=0
# File: /234/LIBERO/libero/lifelong/algos/base.py, line ~145
sed -i 's/persistent_workers=True/persistent_workers=False/' /234/LIBERO/libero/lifelong/algos/base.py
```

### Step 8: Download LIBERO-Spatial Dataset
```bash
cd /234/LIBERO
python benchmark_scripts/download_libero_datasets.py \
  --datasets libero_spatial \
  --download-dir /234/data/datasets
```

### Activate Environment (every new session)
```bash
source /234/miniconda/bin/activate libero
```

---

## 4. Phase 1: BC Baseline Training

### Run Training
```bash
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

### Key Hyperparameters
| Parameter | Value |
|-----------|-------|
| Policy | BCTransformerPolicy (ResNet-18 + Transformer) |
| Optimizer | AdamW |
| Learning Rate | 1e-4 cosine decay to 1e-5 |
| Batch Size | 32 |
| Epochs | 50 per task |
| Eval Frequency | Every 5 epochs, 20 rollouts |
| Demos | 50 per task (500 total) |
| Image Size | 128×128 |
| Sequence Length | 10 frames |
| Action Head | GMM (5 modes) |
| Seed | 42 |

### Model Checkpoints
```
/234/LIBERO/experiments/LIBERO_SPATIAL/Sequential/BCTransformerPolicy_seed42/run_003/
└── task{0-9}_model.pth
```

### BC Baseline Results (run_003)
| Task | Description | BC Success |
|------|-------------|-----------|
| 0 | Between plate & ramekin | 80% |
| 1 | Next to ramekin | 80% |
| 2 | Table center | 100% |
| 3 | On cookie box | 90% |
| 4 | Top drawer | 75% |
| 5 | On ramekin | 70% |
| 6 | Next to cookie box | 85% |
| 7 | On stove | 90% |
| 8 | Next to plate | 80% |
| 9 | On wooden cabinet | 65% |
| **AVG** | | **81.5%** |

---

## 5. Phase 2: Latent Waypoint Extraction

### Run
```bash
source /234/miniconda/bin/activate libero
python3 /234/scripts/visualization/extract_waypoints.py
```

### What It Does
1. Loads each task's frozen BC encoder
2. Runs all 50 demos through `spatial_encode` + `temporal_encode`
3. Extracts 64-dim latent vectors per timestep
4. K-means clusters (k=3, 5, 10) to find waypoints
5. Re-orders clusters by mean temporal position

### Outputs
```
/234/outputs/waypoints/
├── task{0-9}_latents.npy       # Raw latents (N, 64) — not in repo (large)
└── task{0-9}_waypoints.pt      # K-means centers for k=3,5,10
```

---

## 6. Phase 3: Waypoint Visualization & Quality Evaluation

### Run
```bash
python3 /234/scripts/visualization/visualize_waypoints.py   # t-SNE + timeline
python3 /234/scripts/evaluate_waypoints.py                  # silhouette scores
```

### Silhouette Scores (k=5)
| Metric | Value |
|--------|-------|
| Mean | 0.227 |
| Best | Task 6: 0.292 |
| Worst | Task 8: 0.179 |

---

## 7. Phase 4: Residual PPO Fine-Tuning

### Architecture
- Frozen BC policy provides base 7-dim actions
- PPO learns 7-dim residual correction
- Final action = `clip(bc_action + 0.05 * residual, -1, 1)`
- Observation space: 64-dim latent from frozen BC encoder
- Policy/value network: MLP [256, 256]

### Dense Reward Function
```
reward = (prev_dist - curr_dist) * 2.0    # waypoint progress (every step)
       + 0.5  if dist < 0.15              # waypoint reached bonus
       + 1.0  if task_success             # task completion bonus
```

### Key Hyperparameters
| Parameter | Value |
|-----------|-------|
| RESIDUAL_SCALE | 0.05 |
| K (waypoints) | 5 |
| MAX_STEPS | 200 |
| PPO_STEPS | 500,000 |
| learning_rate | 3e-4 |
| n_steps | 2048 |
| batch_size | 64 |
| n_epochs | 10 |
| gamma | 0.99 |
| gae_lambda | 0.95 |
| clip_range | 0.2 |
| ent_coef | 0.01 |

### Run Single Task
```bash
source /234/miniconda/bin/activate libero
tmux new -s ppo_task4
python3 /234/scripts/residual_ppo_single.py --task 4
```

### Run Multiple Tasks in Parallel
```bash
tmux new -s ppo_batch
source /234/miniconda/bin/activate libero
python3 /234/scripts/residual_ppo_single.py --task 0 > /234/outputs/ppo_logs/task0.log 2>&1 &
python3 /234/scripts/residual_ppo_single.py --task 1 > /234/outputs/ppo_logs/task1.log 2>&1 &
python3 /234/scripts/residual_ppo_single.py --task 5 > /234/outputs/ppo_logs/task5.log 2>&1 &
python3 /234/scripts/residual_ppo_single.py --task 9 > /234/outputs/ppo_logs/task9.log 2>&1 &
wait && echo "Done!"
# Detach: Ctrl+B then D
# Reattach: tmux attach -t ppo_batch
```

### Monitor Training
```bash
# Progress across all tasks
grep "episode_reward" /234/outputs/ppo_logs/task*.log | tail -10

# Live tail specific task
tail -f /234/outputs/ppo_logs/task4.log

# Check which processes are running
ps aux | grep residual_ppo_single | grep -v grep

# GPU / RAM usage
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv && free -h
```

### Kill All PPO Runs
```bash
kill $(ps aux | grep residual_ppo_single | grep -v grep | awk '{print $2}')
```

### Model Output Location
```
/234/outputs/ppo/task{id}/
├── best_model/best_model.zip    # Best checkpoint by eval reward
├── final_model.zip
├── checkpoints/
├── eval_logs/
├── tb_logs/
└── monitor.monitor.csv
```

---

## 8. Phase 5: Evaluation

### Evaluate Residual PPO (all tasks, save output)
```bash
source /234/miniconda/bin/activate libero
python3 /234/scripts/eval_ppo.py 2>&1 | tee /234/outputs/eval_$(date +%Y%m%d_%H%M%S).txt
```

### Evaluate Vanilla PPO Ablation
```bash
python3 /234/scripts/eval_vanilla_ppo.py 2>&1 | tee /234/outputs/eval_vanilla_$(date +%Y%m%d_%H%M%S).txt
```

### View All Saved Eval Results
```bash
ls /234/outputs/eval_*.txt
grep "Task\|Average" /234/outputs/eval_*.txt
```

---

## 9. Vanilla PPO Ablation

Vanilla PPO with raw proprio+object state (no BC, sparse reward only) — used to show dense reward is essential.

### Run All 10 Tasks in Parallel
```bash
tmux new -s ppo_vanilla
source /234/miniconda/bin/activate libero
for i in 0 1 2 3 4 5 6 7 8 9; do
  python3 /234/scripts/vanilla_ppo_single.py --task $i > /234/outputs/ppo_logs/vanilla_task${i}.log 2>&1 &
done
wait && echo "Done!"
```

---

## 10. Links

| Resource | URL |
|----------|-----|
| GitHub Repo | https://github.com/leejason2025/DensePPO |
| HuggingFace Models & Data | https://huggingface.co/datasets/leejason2025/DensePPO |

---

## 11. Common Issues & Fixes

| Issue | Fix |
|-------|-----|
| `persistent_workers` crash | Set to `False` in `libero/lifelong/algos/base.py` |
| h5py pickle error during eval | Add `eval.use_mp=false` to training command |
| PyTorch too old for server kernel | Upgrade to `torch==2.0.1+cu118` |
| `get_task_embs` crashes | Pass `[task.language]` not `[task]` |
| BC action shape (10,7) not (7,) | Use `dist.sample()[0, -1]` not `[0]` |
| `shimmy` not found | `pip install shimmy>=0.2.1` |
| `gymnasium` version conflict | `pip install gymnasium==0.29.1` |
| `sklearn` not found | `pip install scikit-learn` |
| GitHub push 403 error | Token missing — re-set remote URL with token |
| `screen` not found | Use `tmux` instead |
| Process killed when terminal closes | Always use `tmux` for long-running jobs |
| conda not active after reconnect | `source /234/miniconda/bin/activate libero` |
| `git: dubious ownership` | `git config --global --add safe.directory /234` |
