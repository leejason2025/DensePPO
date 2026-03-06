"""
Residual PPO Fine-Tuning for LIBERO-Spatial
- Frozen BC policy provides base actions
- PPO learns residual corrections
- Dense reward from latent waypoint progress
- Runs Task 2 (table center) and Task 4 (top drawer)
"""
import os, sys, numpy as np, torch, gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

sys.path.insert(0, "/234/LIBERO")

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.lifelong.models.bc_transformer_policy import BCTransformerPolicy
from libero.lifelong.datasets import get_dataset
from libero.lifelong.utils import get_task_embs

# ── Config ────────────────────────────────────────────────────────────────────
EXP_DIR   = "/234/LIBERO/experiments/LIBERO_SPATIAL/Sequential/BCTransformerPolicy_seed42/run_003"
WP_DIR    = "/234/outputs/waypoints"
OUT_DIR   = "/234/outputs/ppo"
BDDL_DIR  = "/234/LIBERO/libero/libero/bddl_files/libero_spatial"
DATA_DIR  = "/234/data/datasets/libero_spatial"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

K               = 5
SEQ_LEN         = 10
MAX_STEPS       = 200
WAYPOINT_THRESH = 0.15
WAYPOINT_BONUS  = 0.5
TASK_BONUS      = 1.0
RESIDUAL_SCALE  = 0.05
PPO_STEPS       = 500_000

TASK_IDS = [4]
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
task_suite     = benchmark_dict["libero_spatial"]()


def load_bc_policy(task_id):
    ckpt = torch.load(os.path.join(EXP_DIR, f"task{task_id}_model.pth"), map_location=DEVICE)
    cfg  = ckpt["cfg"]

    hdf5_path = os.path.join(DATA_DIR, f"{TASK_NAMES[task_id]}_demo.hdf5")
    _, shape_meta = get_dataset(
        dataset_path=hdf5_path,
        obs_modality=cfg.data.obs.modality,
        initialize_obs_utils=(task_id == TASK_IDS[0]),
        seq_len=cfg.data.seq_len,
    )

    task     = task_suite.get_task(task_id)
    task_emb = get_task_embs(cfg, [task.language])[0].to(DEVICE)

    policy = BCTransformerPolicy(cfg, shape_meta)
    policy.load_state_dict(ckpt["state_dict"])
    policy.to(DEVICE)
    policy.eval()
    for p in policy.parameters():
        p.requires_grad = False

    return policy, task_emb, cfg


def load_waypoints(task_id):
    results = torch.load(os.path.join(WP_DIR, f"task{task_id}_waypoints.pt"))
    return results[K]["centers"]  # (K, 64)


class LiberoResidualEnv(gym.Env):
    def __init__(self, task_id, bc_policy, task_emb, waypoints):
        super().__init__()
        self.task_id   = task_id
        self.bc_policy = bc_policy
        self.task_emb  = task_emb
        self.waypoints = waypoints
        self.K         = len(waypoints)

        task     = task_suite.get_task(task_id)
        bddl     = os.path.join(BDDL_DIR, task.bddl_file)
        self.env = OffScreenRenderEnv(
            bddl_file_name=bddl,
            camera_heights=128,
            camera_widths=128,
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(64,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )

        self.obs_buffer = None
        self.step_count = 0
        self.wp_index   = 0
        self.prev_dist  = None

    def _update_buffer(self, obs):
        av  = torch.tensor(obs["agentview_image"],          dtype=torch.float32).permute(2,0,1) / 255.0
        eh  = torch.tensor(obs["robot0_eye_in_hand_image"], dtype=torch.float32).permute(2,0,1) / 255.0
        jp  = torch.tensor(obs["robot0_joint_pos"],         dtype=torch.float32)
        gp  = torch.tensor(obs["robot0_gripper_qpos"],      dtype=torch.float32)

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
            data = self._make_data_dict()
            dist = self.bc_policy(data)
            action = dist.sample()[0, -1].cpu().numpy()  # last timestep -> (7,)
        return np.clip(action, -1.0, 1.0)

    def _waypoint_reward(self, z):
        target = self.waypoints[self.wp_index]
        z_norm = z / (np.linalg.norm(z) + 1e-8)
        dist   = np.linalg.norm(z_norm - target)

        reward = 0.0
        if self.prev_dist is not None:
            reward += (self.prev_dist - dist) * 2.0
        self.prev_dist = dist

        if dist < WAYPOINT_THRESH and self.wp_index < self.K - 1:
            reward += WAYPOINT_BONUS
            self.wp_index += 1
            self.prev_dist = None

        return reward

    def reset(self):
        raw_obs         = self.env.reset()
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
            dense_reward += TASK_BONUS
            done = True
        if self.step_count >= MAX_STEPS:
            done = True

        info["success"] = sparse_reward > 0
        return z, dense_reward, done, info

    def close(self):
        self.env.close()


def train_task(task_id):
    print(f"\n{'='*60}")
    print(f"Training PPO — Task {task_id}: {TASK_NAMES[task_id][:50]}...")
    print(f"{'='*60}")

    task_out = os.path.join(OUT_DIR, f"task{task_id}")
    os.makedirs(task_out, exist_ok=True)

    bc_policy, task_emb, cfg = load_bc_policy(task_id)
    waypoints = load_waypoints(task_id)
    print(f"  Waypoints shape: {waypoints.shape}")

    def make_env():
        return LiberoResidualEnv(task_id, bc_policy, task_emb, waypoints)

    vec_env  = DummyVecEnv([make_env])
    vec_env  = VecMonitor(vec_env, filename=os.path.join(task_out, "monitor"))
    eval_env = DummyVecEnv([make_env])
    eval_env = VecMonitor(eval_env)

    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(task_out, "best_model"),
            log_path=os.path.join(task_out, "eval_logs"),
            eval_freq=10_000,
            n_eval_episodes=20,
            deterministic=True,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=50_000,
            save_path=os.path.join(task_out, "checkpoints"),
            name_prefix="ppo",
            verbose=1,
        ),
    ]

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=os.path.join(task_out, "tb_logs"),
        device=DEVICE,
        policy_kwargs=dict(net_arch=[256, 256]),
    )

    print(f"  Starting PPO for {PPO_STEPS:,} steps...")
    model.learn(total_timesteps=PPO_STEPS, callback=callbacks, progress_bar=True)
    model.save(os.path.join(task_out, "final_model"))
    print(f"  Saved to {task_out}/final_model.zip")

    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    for task_id in TASK_IDS:
        train_task(task_id)
    print("\nAll done!")
