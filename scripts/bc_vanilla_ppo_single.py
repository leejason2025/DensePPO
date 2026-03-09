"""BC + Vanilla PPO ablation: residual PPO with sparse reward only (no waypoints)."""
import os, sys, argparse, numpy as np, torch, gym
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

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=int, required=True)
args = parser.parse_args()
TASK_ID = args.task

EXP_DIR  = "/234/LIBERO/experiments/LIBERO_SPATIAL/Sequential/BCTransformerPolicy_seed42/run_003"
BDDL_DIR = "/234/LIBERO/libero/libero/bddl_files/libero_spatial"
DATA_DIR = "/234/data/datasets/libero_spatial"
OUT_DIR  = f"/234/outputs/ppo_bc_vanilla/task{TASK_ID}"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

MAX_STEPS      = 200
PPO_STEPS      = 500_000
SEQ_LEN        = 10
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

os.makedirs(OUT_DIR, exist_ok=True)

benchmark_dict = benchmark.get_benchmark_dict()
task_suite     = benchmark_dict["libero_spatial"]()


def load_bc_policy(task_id):
    ckpt = torch.load(os.path.join(EXP_DIR, f"task{task_id}_model.pth"), map_location=DEVICE)
    cfg  = ckpt["cfg"]
    _, shape_meta = get_dataset(
        dataset_path=os.path.join(DATA_DIR, f"{TASK_NAMES[task_id]}_demo.hdf5"),
        obs_modality=cfg.data.obs.modality,
        initialize_obs_utils=True,
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


class BCVanillaEnv(gym.Env):
    """Residual env with BC base policy but SPARSE reward only — no waypoints."""
    def __init__(self, task_id, bc_policy, task_emb):
        super().__init__()
        self.task_id   = task_id
        self.bc_policy = bc_policy
        self.task_emb  = task_emb
        task = task_suite.get_task(task_id)
        self.env = OffScreenRenderEnv(
            bddl_file_name=os.path.join(BDDL_DIR, task.bddl_file),
            camera_heights=128, camera_widths=128,
        )
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(64,), dtype=np.float32)
        self.action_space      = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.obs_buffer = None
        self.step_count = 0

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

    def reset(self):
        raw_obs = self.env.reset()
        self.obs_buffer = None
        self.step_count = 0
        self._update_buffer(raw_obs)
        return self._get_latent()

    def step(self, residual_action):
        bc_action = self._get_bc_action()
        action    = np.clip(bc_action + RESIDUAL_SCALE * residual_action, -1.0, 1.0)
        raw_obs, sparse_reward, done, info = self.env.step(action)
        self._update_buffer(raw_obs)
        self.step_count += 1
        z = self._get_latent()
        # SPARSE REWARD ONLY — no waypoint shaping
        reward = sparse_reward
        if sparse_reward > 0:
            done = True
        if self.step_count >= MAX_STEPS:
            done = True
        info["success"] = sparse_reward > 0
        return z, reward, done, info

    def close(self):
        self.env.close()


# ── Train ─────────────────────────────────────────────────────────────────────
print(f"Task {TASK_ID}: BC + Vanilla PPO (sparse reward only)...")
bc_policy, task_emb, cfg = load_bc_policy(TASK_ID)

def make_env():
    return BCVanillaEnv(TASK_ID, bc_policy, task_emb)

vec_env  = DummyVecEnv([make_env])
vec_env  = VecMonitor(vec_env, filename=os.path.join(OUT_DIR, "monitor"))
eval_env = DummyVecEnv([make_env])
eval_env = VecMonitor(eval_env)

callbacks = [
    EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(OUT_DIR, "best_model"),
        log_path=os.path.join(OUT_DIR, "eval_logs"),
        eval_freq=10_000,
        n_eval_episodes=20,
        deterministic=True,
        verbose=1,
    ),
    CheckpointCallback(
        save_freq=100_000,
        save_path=os.path.join(OUT_DIR, "checkpoints"),
        name_prefix="bc_vanilla_ppo",
        verbose=0,
    ),
]

model = PPO(
    "MlpPolicy", vec_env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=1,
    tensorboard_log=os.path.join(OUT_DIR, "tb_logs"),
    device=DEVICE,
    policy_kwargs=dict(net_arch=[256, 256]),
)

model.learn(total_timesteps=PPO_STEPS, callback=callbacks, progress_bar=False)
model.save(os.path.join(OUT_DIR, "final_model"))
print(f"Task {TASK_ID}: done!")
vec_env.close()
eval_env.close()
