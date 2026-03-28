"""
Configurable DQN baseline for highway-env.

This keeps the original Leurent-style highway-v0 setup, but adds CLI
configuration, per-run artifact folders, and TensorBoard logging so we can
run repeatable sweeps and keep the outputs organized.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401 - registers highway environments
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "dqn_highway_sweep"


def make_config() -> dict:
    return {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 5,
            "features": ["presence", "x", "y", "vx", "vy"],
            "absolute": False,
        },
        "action": {
            "type": "DiscreteMetaAction",
        },
        "lanes_count": 3,
        "vehicles_count": 20,
        "duration": 40,
    }


def make_env(render_mode: str = "rgb_array"):
    return gym.make("highway-v0", render_mode=render_mode, config=make_config())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the baseline Leurent DQN on highway-v0")
    parser.add_argument("--timesteps", type=int, default=200000, help="Total training timesteps")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="DQN learning rate")
    parser.add_argument("--buffer-size", type=int, default=15000, help="Replay buffer size")
    parser.add_argument("--learning-starts", type=int, default=200, help="Warmup steps before gradient updates")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--gamma", type=float, default=0.8, help="Discount factor")
    parser.add_argument(
        "--target-update-interval",
        type=int,
        default=50,
        help="Target network update interval",
    )
    parser.add_argument("--seed", type=int, default=42, help="Training seed")
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of parallel highway environments for rollout collection",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device passed to SB3 (default: auto)",
    )
    parser.add_argument("--run-name", default="baseline", help="Run name for logs and result folders")
    parser.add_argument(
        "--results-root",
        default=str(DEFAULT_RESULTS_ROOT),
        help="Root directory for run artifacts",
    )
    parser.add_argument("--verbose", type=int, default=1, help="SB3 verbosity level")
    return parser.parse_args()


def train_and_evaluate(args: argparse.Namespace) -> dict:
    results_root = Path(args.results_root).resolve()
    run_dir = results_root / args.run_name
    models_dir = run_dir / "models"
    tb_dir = run_dir / "tensorboard"
    models_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    if args.num_envs < 1:
        raise ValueError("--num-envs must be >= 1")

    env_kwargs = {"render_mode": "rgb_array", "config": make_config()}
    if args.num_envs == 1:
        env = make_vec_env(
            "highway-v0",
            n_envs=1,
            seed=args.seed,
            env_kwargs=env_kwargs,
        )
    else:
        env = make_vec_env(
            "highway-v0",
            n_envs=args.num_envs,
            seed=args.seed,
            env_kwargs=env_kwargs,
            vec_env_cls=SubprocVecEnv,
        )

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        target_update_interval=args.target_update_interval,
        tensorboard_log=str(tb_dir),
        seed=args.seed,
        device=args.device,
        verbose=args.verbose,
    )

    print(f"Starting training for {args.timesteps} timesteps...")
    model.learn(total_timesteps=args.timesteps, tb_log_name=args.run_name)

    model_path = models_dir / "elurant_dqn"
    model.save(str(model_path))
    print(f"Model saved to {model_path}.zip")

    eval_env = Monitor(make_env(render_mode="rgb_array"))
    eval_env.reset(seed=args.seed + 1000)
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=args.eval_episodes,
    )
    print(
        f"Evaluation over {args.eval_episodes} episodes: "
        f"mean reward = {mean_reward:.2f}, std = {std_reward:.2f}"
    )

    summary = {
        "run_name": args.run_name,
        "timesteps": args.timesteps,
        "eval_episodes": args.eval_episodes,
        "learning_rate": args.learning_rate,
        "buffer_size": args.buffer_size,
        "learning_starts": args.learning_starts,
        "batch_size": args.batch_size,
        "gamma": args.gamma,
        "target_update_interval": args.target_update_interval,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "device": args.device,
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "model_path": str(model_path.with_suffix(".zip")),
        "tensorboard_dir": str(tb_dir / args.run_name),
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary written to {summary_path}")

    eval_env.close()
    env.close()
    return summary


def main() -> None:
    args = parse_args()
    train_and_evaluate(args)


if __name__ == "__main__":
    main()
