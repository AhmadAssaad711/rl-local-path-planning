from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[5]
PROJECT_VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"


def maybe_reexec_with_project_venv() -> None:
    if os.environ.get("PPO_REWARD_STUDY_SKIP_VENV_REEXEC") == "1":
        return

    if not PROJECT_VENV_PYTHON.exists():
        return

    try:
        import gymnasium  # noqa: F401
        import stable_baselines3  # noqa: F401
        return
    except ModuleNotFoundError:
        current_python = Path(sys.executable).resolve()
        venv_python = PROJECT_VENV_PYTHON.resolve()
        if current_python == venv_python:
            raise

        result = subprocess.run(
            [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]],
            check=False,
        )
        raise SystemExit(result.returncode)


maybe_reexec_with_project_venv()

from stable_baselines3 import PPO

from study_env import make_study_env


SCRIPT_DIR = Path(__file__).resolve().parent
TRIAL_DIR = SCRIPT_DIR / "m" / "t25_crm150_hs060_rl000"
MODEL_PATH = TRIAL_DIR / "ppo_highway.zip"
MANIFEST_PATH = TRIAL_DIR / "trial_config.json"
N_EPISODES = 10
SEED_OFFSET = 1000
PAUSE_BETWEEN_EPISODES_SEC = 0.5


def load_reward_config(manifest_path: Path) -> dict:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return dict(payload["reward_config"])


def main() -> None:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing trial manifest: {MANIFEST_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing PPO model: {MODEL_PATH}")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    reward_config = load_reward_config(MANIFEST_PATH)
    base_seed = int(manifest["seed"])

    env = make_study_env(render_mode="human", reward_config=reward_config)
    model = PPO.load(str(MODEL_PATH), env=env)

    print(f"Loaded model: {MODEL_PATH}")
    print(f"Trial: {manifest['run_name']}")
    print(f"Reward config: {reward_config}")
    print(f"Running {N_EPISODES} visual episodes...")

    try:
        for episode_idx in range(N_EPISODES):
            obs, info = env.reset(seed=base_seed + SEED_OFFSET + episode_idx)
            env.render()

            total_reward = 0.0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                env.render()

            metrics = dict(info.get("episode_metrics", {}))
            print(
                f"[episode {episode_idx + 1}/{N_EPISODES}] "
                f"reward={total_reward:.3f} "
                f"collision={metrics.get('collision', 0.0)} "
                f"offroad={metrics.get('offroad', 0.0)} "
                f"speed={metrics.get('mean_forward_speed', 0.0):.3f} "
                f"right_lane={metrics.get('right_lane_ratio', 0.0):.3f}"
            )
            time.sleep(PAUSE_BETWEEN_EPISODES_SEC)
    finally:
        env.close()


if __name__ == "__main__":
    main()
