"""Minimal Stable-Baselines3 training entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..env.core import UnstructuredTrafficEnv


def build_model(algo: str, env, timesteps: int):
    try:
        if algo == "ppo":
            from stable_baselines3 import PPO

            return PPO("MlpPolicy", env, verbose=1, n_steps=256, batch_size=64, learning_rate=3e-4)
        from stable_baselines3 import DQN

        return DQN(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            buffer_size=max(10_000, timesteps // 2),
            learning_starts=500,
            batch_size=64,
            gamma=0.99,
            target_update_interval=250,
        )
    except ImportError as exc:
        raise SystemExit("stable_baselines3 is required for this script.") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an SB3 agent on the unstructured traffic environment.")
    parser.add_argument("--scenario", default="dense_urban_chaos")
    parser.add_argument("--algo", choices=["dqn", "ppo"], default="ppo")
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="models")
    args = parser.parse_args()

    env = UnstructuredTrafficEnv(args.scenario, render_mode=None)
    model = build_model(args.algo, env, args.timesteps)
    model.learn(total_timesteps=args.timesteps, progress_bar=False)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"unstructured_{args.algo}_{args.scenario}"
    model.save(model_path)
    env.close()
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
