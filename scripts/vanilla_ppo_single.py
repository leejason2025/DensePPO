"""Vanilla PPO with no BC policy — pure RL from scratch for ablation."""
import os, sys, argparse, numpy as np, torch, gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

sys.path.insert(0, "/234/LIBERO")

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=int, required=True)
args = parser.parse_args()
TASK_ID = args.task

BDDL_DIR  = "/234/LIBERO/libero/libero/bddl_files/libero_spatial"
OUT_DIR   = f"/234/outputs/ppo_vanilla/task{TASK_ID}"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
MAX_STEPS = 200
PPO_STEPS = 500_000

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


class LiberoVanillaEnv(gym.Env):
    """Plain LIBERO env with raw obs vector — no BC policy, no waypoints."""
    def __init__(self, task_id):
        super().__init__()
        task = task_suite.get_task(task_id)
        self.env = OffScreenRenderEnv(
            bddl_file_name=os.path.join(BDDL_DIR, task.bddl_file),
            camera_heights=128,
            camera_widths=128,
        )
        # use proprio state only (no images) for MlpPolicy
        # robot0_proprio-state is 39-dim, object-state is 70-dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(109,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )
        self.step_count = 0

    def _get_obs(self, raw_obs):
        proprio = np.array(raw_obs["robot0_proprio-state"], dtype=np.float32)
        obj     = np.array(raw_obs["object-state"],         dtype=np.float32)
        return np.concatenate([proprio, obj])

    def reset(self):
        raw_obs     = self.env.reset()
        self.step_count = 0
        return self._get_obs(raw_obs)

    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        self.step_count += 1
        if self.step_count >= MAX_STEPS:
            done = True
        info["success"] = reward > 0
        return self._get_obs(raw_obs), reward, done, info

    def close(self):
        self.env.close()


# ── Train ─────────────────────────────────────────────────────────────────────
print(f"Task {TASK_ID}: starting vanilla PPO...")

def make_env():
    return LiberoVanillaEnv(TASK_ID)

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
        name_prefix="ppo_vanilla",
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

print(f"Task {TASK_ID}: training for {PPO_STEPS:,} steps...")
model.learn(total_timesteps=PPO_STEPS, callback=callbacks, progress_bar=False)
model.save(os.path.join(OUT_DIR, "final_model"))
print(f"Task {TASK_ID}: done!")

vec_env.close()
eval_env.close()
