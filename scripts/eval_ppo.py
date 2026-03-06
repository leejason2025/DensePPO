import os, sys, numpy as np, torch
sys.path.insert(0, '/234/LIBERO')

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.lifelong.models.bc_transformer_policy import BCTransformerPolicy
from libero.lifelong.datasets import get_dataset
from libero.lifelong.utils import get_task_embs
from stable_baselines3 import PPO
from gym import spaces
import gym

EXP_DIR  = "/234/LIBERO/experiments/LIBERO_SPATIAL/Sequential/BCTransformerPolicy_seed42/run_003"
WP_DIR   = "/234/outputs/waypoints"
BDDL_DIR = "/234/LIBERO/libero/libero/bddl_files/libero_spatial"
DATA_DIR = "/234/data/datasets/libero_spatial"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
K        = 5
SEQ_LEN  = 10
MAX_STEPS = 200
RESIDUAL_SCALE = 0.05

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

BC_BASELINE = {0: 80, 1: 80, 2: 100, 3: 90, 4: 75, 5: 70, 6: 85, 7: 90, 8: 80, 9: 65}

benchmark_dict = benchmark.get_benchmark_dict()
task_suite     = benchmark_dict["libero_spatial"]()

def load_bc_policy(task_id):
    ckpt = torch.load(os.path.join(EXP_DIR, f"task{task_id}_model.pth"), map_location=DEVICE)
    cfg  = ckpt["cfg"]
    _, shape_meta = get_dataset(
        dataset_path=os.path.join(DATA_DIR, f"{TASK_NAMES[task_id]}_demo.hdf5"),
        obs_modality=cfg.data.obs.modality,
        initialize_obs_utils=(task_id == 0),
        seq_len=cfg.data.seq_len,
    )
    task     = task_suite.get_task(task_id)
    task_emb = get_task_embs(cfg, [task.language])[0].to(DEVICE)
    policy   = BCTransformerPolicy(cfg, shape_meta)
    policy.load_state_dict(ckpt["state_dict"])
    policy.to(DEVICE)
    policy.eval()
    for p in policy.parameters():
        p.requires_grad = False
    return policy, task_emb, cfg

def load_waypoints(task_id):
    results = torch.load(os.path.join(WP_DIR, f"task{task_id}_waypoints.pt"))
    return results[K]["centers"]

class LiberoResidualEnv(gym.Env):
    def __init__(self, task_id, bc_policy, task_emb, waypoints):
        super().__init__()
        self.task_id   = task_id
        self.bc_policy = bc_policy
        self.task_emb  = task_emb
        self.waypoints = waypoints
        self.K         = len(waypoints)
        task = task_suite.get_task(task_id)
        self.env = OffScreenRenderEnv(
            bddl_file_name=os.path.join(BDDL_DIR, task.bddl_file),
            camera_heights=128, camera_widths=128,
        )
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(64,), dtype=np.float32)
        self.action_space      = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.obs_buffer = None
        self.step_count = 0
        self.wp_index   = 0
        self.prev_dist  = None

    def _update_buffer(self, obs):
        av = torch.tensor(obs["agentview_image"],          dtype=torch.float32).permute(2,0,1) / 255.0
        eh = torch.tensor(obs["robot0_eye_in_hand_image"], dtype=torch.float32).permute(2,0,1) / 255.0
        jp = torch.tensor(obs["robot0_joint_pos"],         dtype=torch.float32)
        gp = torch.tensor(obs["robot0_gripper_qpos"],      dtype=torch.float32)
        if self.obs_buffer is None:
            self.obs_buffer = {
                "av": av.unsqueeze(0).repeat(SEQ_LEN,1,1,1),
                "eh": eh.unsqueeze(0).repeat(SEQ_LEN,1,1,1),
                "jp": jp.unsqueeze(0).repeat(SEQ_LEN,1),
                "gp": gp.unsqueeze(0).repeat(SEQ_LEN,1),
            }
        else:
            self.obs_buffer["av"] = torch.cat([self.obs_buffer["av"][1:], av.unsqueeze(0)], dim=0)
            self.obs_buffer["eh"] = torch.cat([self.obs_buffer["eh"][1:], eh.unsqueeze(0)], dim=0)
            self.obs_buffer["jp"] = torch.cat([self.obs_buffer["jp"][1:], jp.unsqueeze(0)], dim=0)
            self.obs_buffer["gp"] = torch.cat([self.obs_buffer["gp"][1:], gp.unsqueeze(0)], dim=0)

    def _make_data_dict(self):
        return {
            "obs": {
                "agentview_rgb":   self.obs_buffer["av"].unsqueeze(0).to(DEVICE),
                "eye_in_hand_rgb": self.obs_buffer["eh"].unsqueeze(0).to(DEVICE),
                "joint_states":    self.obs_buffer["jp"].unsqueeze(0).to(DEVICE),
                "gripper_states":  self.obs_buffer["gp"].unsqueeze(0).to(DEVICE),
            },
            "task_emb": self.task_emb.unsqueeze(0).to(DEVICE),
        }

    def _get_latent(self):
        with torch.no_grad():
            data = self._make_data_dict()
            x = self.bc_policy.spatial_encode(data)
            x = self.bc_policy.temporal_encode(x)
            return x[0, -1].cpu().numpy().astype(np.float32)

    def _get_bc_action(self):
        with torch.no_grad():
            dist   = self.bc_policy(self._make_data_dict())
            action = dist.sample()[0, -1].cpu().numpy()
        return np.clip(action, -1.0, 1.0)

    def _waypoint_reward(self, z):
        target = self.waypoints[self.wp_index]
        z_norm = z / (np.linalg.norm(z) + 1e-8)
        dist   = np.linalg.norm(z_norm - target)
        reward = 0.0
        if self.prev_dist is not None:
            reward += (self.prev_dist - dist) * 2.0
        self.prev_dist = dist
        if dist < 0.15 and self.wp_index < self.K - 1:
            reward += 0.5
            self.wp_index += 1
            self.prev_dist = None
        return reward

    def reset(self):
        raw_obs = self.env.reset()
        self.obs_buffer = None
        self.step_count = 0
        self.wp_index   = 0
        self.prev_dist  = None
        self._update_buffer(raw_obs)
        return self._get_latent()

    def step(self, residual_action):
        bc_action = self._get_bc_action()
        action    = np.clip(bc_action + RESIDUAL_SCALE * residual_action, -1.0, 1.0)
        raw_obs, sparse_reward, done, info = self.env.step(action)
        self._update_buffer(raw_obs)
        self.step_count += 1
        z            = self._get_latent()
        dense_reward = self._waypoint_reward(z)
        if sparse_reward > 0:
            dense_reward += 1.0
            done = True
        if self.step_count >= MAX_STEPS:
            done = True
        info["success"] = sparse_reward > 0
        return z, dense_reward, done, info

    def close(self):
        self.env.close()


# ── Eval ─────────────────────────────────────────────────────────────────────
N_EVAL   = 20
EVAL_TASKS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # tasks with trained PPO models

print("\n" + "="*55)
print("PPO Evaluation Results")
print("="*55)

results = {}
for task_id in EVAL_TASKS:
    print(f"\nTask {task_id}...", flush=True)
    bc_policy, task_emb, cfg = load_bc_policy(task_id)
    waypoints = load_waypoints(task_id)
    env   = LiberoResidualEnv(task_id, bc_policy, task_emb, waypoints)
    model = PPO.load(f'/234/outputs/ppo/task{task_id}/best_model/best_model.zip')

    successes = []
    for ep in range(N_EVAL):
        obs  = env.reset()
        done = False
        success = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            if info.get('success'):
                success = True
        successes.append(success)
        print(f"  ep {ep+1:2d}: {'✓' if success else '✗'}", flush=True)

    rate = np.mean(successes) * 100
    results[task_id] = rate
    bc = BC_BASELINE[task_id]
    print(f"  BC: {bc}%  →  PPO: {rate:.1f}%  ({rate-bc:+.1f}%)")
    env.close()

print("\n" + "="*55)
print("SUMMARY")
print("="*55)
for task_id in EVAL_TASKS:
    bc  = BC_BASELINE[task_id]
    ppo = results[task_id]
    print(f"  Task {task_id}: {bc}% → {ppo:.1f}% ({ppo-bc:+.1f}%)")
avg_bc  = np.mean([BC_BASELINE[t] for t in EVAL_TASKS])
avg_ppo = np.mean([results[t] for t in EVAL_TASKS])
print(f"  Average: {avg_bc:.1f}% → {avg_ppo:.1f}% ({avg_ppo-avg_bc:+.1f}%)")
