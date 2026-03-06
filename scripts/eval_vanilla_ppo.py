import os, sys, numpy as np, torch, gym
from gym import spaces
from stable_baselines3 import PPO

sys.path.insert(0, "/234/LIBERO")
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

BDDL_DIR = "/234/LIBERO/libero/libero/bddl_files/libero_spatial"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
MAX_STEPS = 200
N_EVAL    = 20

BC_BASELINE = {0: 80, 1: 80, 2: 100, 3: 90, 4: 75, 5: 70, 6: 85, 7: 90, 8: 80, 9: 65}

benchmark_dict = benchmark.get_benchmark_dict()
task_suite     = benchmark_dict["libero_spatial"]()

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

class LiberoVanillaEnv(gym.Env):
    def __init__(self, task_id):
        super().__init__()
        task = task_suite.get_task(task_id)
        self.env = OffScreenRenderEnv(
            bddl_file_name=os.path.join(BDDL_DIR, task.bddl_file),
            camera_heights=128, camera_widths=128,
        )
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(109,), dtype=np.float32)
        self.action_space      = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.step_count = 0

    def _get_obs(self, raw_obs):
        proprio = np.array(raw_obs["robot0_proprio-state"], dtype=np.float32)
        obj     = np.array(raw_obs["object-state"],         dtype=np.float32)
        return np.concatenate([proprio, obj])

    def reset(self):
        self.step_count = 0
        return self._get_obs(self.env.reset())

    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        self.step_count += 1
        if self.step_count >= MAX_STEPS:
            done = True
        info["success"] = reward > 0
        return self._get_obs(raw_obs), reward, done, info

    def close(self):
        self.env.close()


print("\n" + "="*55)
print("Vanilla PPO Evaluation Results")
print("="*55)

results = {}
for task_id in range(10):
    print(f"\nTask {task_id}...", flush=True)
    env   = LiberoVanillaEnv(task_id)
    model = PPO.load(f'/234/outputs/ppo_vanilla/task{task_id}/best_model/best_model.zip')

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
    print(f"  BC: {bc}%  |  Vanilla PPO: {rate:.1f}%")
    env.close()

print("\n" + "="*55)
print("SUMMARY")
print("="*55)
for task_id in range(10):
    bc  = BC_BASELINE[task_id]
    van = results[task_id]
    print(f"  Task {task_id}: BC={bc}%  Vanilla PPO={van:.1f}%")
print(f"\n  BC Average:          {np.mean(list(BC_BASELINE.values())):.1f}%")
print(f"  Vanilla PPO Average: {np.mean(list(results.values())):.1f}%")
