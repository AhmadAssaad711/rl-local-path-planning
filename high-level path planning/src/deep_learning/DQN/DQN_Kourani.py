import time
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

# ==========================================
# 1. PATHS & DIRECTORIES
# ==========================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[4]
PROJECT_VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
MODEL_DIR = PROJECT_ROOT / "logs"
VIDEO_DIR = MODEL_DIR / "videos_ttc"
MODEL_STEM = "model_ttc"
MODEL_PATH = MODEL_DIR / f"{MODEL_STEM}.zip"


def maybe_reexec_with_project_venv() -> None:
    if os.environ.get("KOURANI_DQN_SKIP_VENV_REEXEC") == "1":
        return

    if not PROJECT_VENV_PYTHON.exists():
        return

    try:
        import highway_env  # noqa: F401
        import stable_baselines3  # noqa: F401
        return
    except ModuleNotFoundError:
        current_python = Path(sys.executable).resolve()
        venv_python = PROJECT_VENV_PYTHON.resolve()
        if current_python == venv_python:
            raise

        child_env = dict(os.environ)
        child_env["KOURANI_DQN_SKIP_VENV_REEXEC"] = "1"
        result = subprocess.run(
            [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]],
            check=False,
            env=child_env,
        )
        raise SystemExit(result.returncode)


maybe_reexec_with_project_venv()

from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from ttc_reward_wrapper import DEFAULT_TTC_CONFIG, build_ttc_config, make_ttc_highway_env

# ==========================================
# 2. ENVIRONMENT & TRAINING CONFIG
# ==========================================
ENV_NAME = "highway-v0"
N_ENVS = 24
TOTAL_TIMESTEPS = 100000
TRAIN = True

ENV_CONFIG = {
    "collision_reward": -5.0,
    "high_speed_reward": 0.3,
    "right_lane_reward": 0.15,
    "lane_change_reward": -0.01,
    "reward_speed_range": [20, 30],
}
TTC_CONFIG = build_ttc_config(DEFAULT_TTC_CONFIG)

# ==========================================
# 3. MODEL HYPERPARAMETERS
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
    "tensorboard_log": str(MODEL_DIR),
}


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


LIMIT_BY_EPISODES = False
MAX_EPISODES = 2000


def make_kourani_ttc_env(
    render_mode: str = "rgb_array",
    config: Mapping[str, Any] | None = None,
    ttc_config: Mapping[str, Any] | None = None,
):
    merged_config = dict(ENV_CONFIG)
    if config:
        merged_config.update(dict(config))
    return make_ttc_highway_env(
        render_mode=render_mode,
        config=merged_config,
        ttc_config=build_ttc_config(ttc_config),
    )


def train_highway_dqn(
    custom_env_config: Mapping[str, Any] | None = None,
    custom_model_dir: str | Path | None = None,
    timesteps: int | None = None,
    ttc_config: Mapping[str, Any] | None = None,
    n_envs: int | None = None,
):
    """
    Execute Kourani DQN training with TTC reward shaping.
    """

    current_config = dict(ENV_CONFIG)
    if custom_env_config:
        current_config.update(dict(custom_env_config))

    current_ttc_config = build_ttc_config(ttc_config or TTC_CONFIG)
    current_timesteps = timesteps if timesteps is not None else TOTAL_TIMESTEPS
    current_n_envs = max(int(n_envs) if n_envs is not None else N_ENVS, 1)
    current_model_dir = Path(custom_model_dir).resolve() if custom_model_dir else MODEL_DIR
    current_video_dir = current_model_dir / VIDEO_DIR.name
    current_model_base = current_model_dir / MODEL_STEM
    current_model_path = current_model_dir / f"{MODEL_STEM}.zip"
    current_model_dir.mkdir(parents=True, exist_ok=True)

    print(f"Spawning {current_n_envs} parallel environments for: {ENV_NAME}")
    print(f"Reward config: {current_config}")
    print(f"TTC reward shaping config: {current_ttc_config}")

    train_env = make_vec_env(
        make_kourani_ttc_env,
        n_envs=current_n_envs,
        vec_env_cls=SubprocVecEnv if current_n_envs > 1 else None,
        env_kwargs={
            "render_mode": "rgb_array",
            "config": current_config,
            "ttc_config": current_ttc_config,
        },
    )

    local_params = MODEL_PARAMS.copy()
    local_params["tensorboard_log"] = str(current_model_dir)
    local_params["gradient_steps"] = current_n_envs

    model = DQN("MlpPolicy", train_env, **local_params)

    try:
        if TRAIN:
            print(f"Starting {current_timesteps} step local training...")
            start_time = time.time()
            callback = StopOnEpisodesCallback(max_episodes=MAX_EPISODES, verbose=1) if LIMIT_BY_EPISODES else None

            model.learn(total_timesteps=current_timesteps, callback=callback)
            print(f"[INFO] Training took {time.time() - start_time:.2f} seconds")
            model.save(str(current_model_base))
            del model

        print("Training complete! Loading model and recording evaluation video...")
        eval_env = make_kourani_ttc_env(
            render_mode="rgb_array",
            config=current_config,
            ttc_config=current_ttc_config,
        )
        eval_env.unwrapped.config["simulation_frequency"] = 15

        env = RecordVideo(eval_env, video_folder=str(current_video_dir), episode_trigger=lambda e: True)
        env.unwrapped.set_record_video_wrapper(env)

        model = DQN.load(str(current_model_path), env=env)

        for _ in range(10):
            done = truncated = False
            obs, info = env.reset()
            while not (done or truncated):
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                env.render()

        print(f"Saved TTC-trained model to {current_model_path}")
        print(f"Videos saved to {current_video_dir}")
        env.close()
    finally:
        train_env.close()


if __name__ == "__main__":
    train_highway_dqn()
