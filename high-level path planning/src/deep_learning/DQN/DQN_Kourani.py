import os
import time
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import highway_env  # noqa: F401

# ==========================================
# 🔧 1. PATHS & DIRECTORIES
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "logs")
VIDEO_DIR = os.path.join(MODEL_DIR, "videos")
MODEL_PATH = os.path.join(MODEL_DIR, "model")

# ==========================================
# 🚗 2. ENVIRONMENT & TRAINING CONFIG
# ==========================================
ENV_NAME = "highway-v0"
N_ENVS = 24                # Number of parallel CPU environments to spawn
TOTAL_TIMESTEPS = 100000
TRAIN = True

ENV_CONFIG = {
    "collision_reward": -5.0,    # -1.0      # Increased penalty
    "high_speed_reward": 0.3,    #  0.4      # Reduced to prioritize safety over speed
    "right_lane_reward": 0.15,   #  0.1      # Added
    "lane_change_reward": -0.01,  #  0.0      # Kept at 0
    "reward_speed_range": [20, 30]
}

# ==========================================
# 🧠 3. MODEL HYPERPARAMETERS
# ==========================================
MODEL_PARAMS = {
    "policy_kwargs": dict(net_arch=[256, 256]),
    "learning_rate": 5e-4,
    "buffer_size": 15000,
    "learning_starts": 2000,
    "batch_size": 128,
    "gamma": 0.8,
    "train_freq": 1,
    "gradient_steps": N_ENVS,
    "target_update_interval": 50,
    "verbose": 1,
    "tensorboard_log": MODEL_DIR
}

# ==========================================
# ⏱️ 4. CALLBACKS & EXECUTION
# ==========================================
class StopOnEpisodesCallback(BaseCallback):
    def __init__(self, max_episodes: int, verbose: int = 0):
        super().__init__(verbose)
        self.max_episodes = max_episodes
        self.num_episodes = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        if dones is not None:
            self.num_episodes += int(sum(dones))
        if self.num_episodes >= self.max_episodes:
            if self.verbose > 0:
                print(f"[INFO] Stopping training! Reached {self.num_episodes} episodes.")
            return False
        return True

LIMIT_BY_EPISODES = False  # Set to True to exactly match rl-agents 2000 episodes
MAX_EPISODES = 2000

def train_highway_dqn(custom_env_config=None, custom_model_dir=None, timesteps=None):
    """
    Importable function to execute parallel DQN training sessions with override configs.
    """
    # Override defaults
    current_config = ENV_CONFIG.copy()
    if custom_env_config:
        current_config.update(custom_env_config)
        
    current_timesteps = timesteps if timesteps is not None else TOTAL_TIMESTEPS
    current_model_dir = custom_model_dir if custom_model_dir else MODEL_DIR
    current_video_dir = os.path.join(current_model_dir, "videos")
    current_model_path = os.path.join(current_model_dir, "model")

    print(f"Spawning {N_ENVS} parallel environments for: {ENV_NAME}")

    # Create vectorized environment for parallel processing
    train_env = make_vec_env(
        ENV_NAME, 
        n_envs=N_ENVS, 
        vec_env_cls=SubprocVecEnv,
        env_kwargs={"config": current_config}
    )

    # Scoped parameters to separate TensorBoard graphs
    local_params = MODEL_PARAMS.copy()
    local_params["tensorboard_log"] = current_model_dir

    model = DQN("MlpPolicy", train_env, **local_params)

    if TRAIN:
        print(f"Starting {current_timesteps} step local training...")
        start_time = time.time()
        callback = StopOnEpisodesCallback(max_episodes=MAX_EPISODES, verbose=1) if LIMIT_BY_EPISODES else None
        
        model.learn(total_timesteps=current_timesteps, callback=callback)
        print(f"[INFO] Training took {time.time() - start_time:.2f} seconds")
        model.save(current_model_path)
        del model

    print("Training complete! Loading model and recording evaluation video...")
    eval_env = gym.make(ENV_NAME, render_mode="rgb_array", config=current_config)
    eval_env.unwrapped.config["simulation_frequency"] = 15 
    
    env = RecordVideo(eval_env, video_folder=current_video_dir, episode_trigger=lambda e: True)
    env.unwrapped.set_record_video_wrapper(env)
    
    model = DQN.load(current_model_path, env=env)

    for episodes in range(10):
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            env.render()
            
    print(f"Videos saved to {current_video_dir}")
    env.close()

if __name__ == "__main__":
    train_highway_dqn()
