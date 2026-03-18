"""Train PPO in stages and keep the best checkpoint by custom diagnostics."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO

from elurant_ppo_ablation import (
    ACTION_VARIANTS,
    OBSERVATION_VARIANTS,
    VARIANTS,
    build_model,
    make_env,
    run_diagnostics,
    summarise_episodes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-wise PPO training with custom checkpoint selection.")
    parser.add_argument("--reward-variant", default="progress_clearance", choices=VARIANTS)
    parser.add_argument("--action-variant", default="intent_guarded_shielded", choices=ACTION_VARIANTS)
    parser.add_argument("--observation-variant", default="gap_augmented", choices=OBSERVATION_VARIANTS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=4096)
    parser.add_argument("--stage-steps", type=int, default=512)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", default="experiments/ppo_final")
    return parser.parse_args()


def score_summary(summary: dict[str, float]) -> float:
    return (
        summary["mean_distance_travelled"]
        - 250.0 * summary["crash_rate"]
        - 120.0 * summary["mean_stop_ratio"]
        + 20.0 * summary["mean_lane_changes"]
    )


def main() -> None:
    args = parse_args()
    if args.batch_size > args.n_steps:
        raise ValueError("--batch-size must be <= --n-steps")
    if args.total_timesteps % args.stage_steps != 0:
        raise ValueError("--total-timesteps must be divisible by --stage-steps")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / timestamp / f"seed_{args.seed}"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(
        render_mode=None,
        reward_variant=args.reward_variant,
        action_variant=args.action_variant,
        observation_variant=args.observation_variant,
    )
    env.reset(seed=args.seed)
    model = build_model(env, seed=args.seed, n_steps=args.n_steps, batch_size=args.batch_size)

    stages: list[dict[str, object]] = []
    best_score = float("-inf")
    best_path: str | None = None
    steps_done = 0

    while steps_done < args.total_timesteps:
        model.learn(total_timesteps=args.stage_steps, progress_bar=False, reset_num_timesteps=False)
        steps_done += args.stage_steps

        checkpoint_base = checkpoint_dir / f"step_{steps_done}"
        model.save(checkpoint_base)
        diagnostics = run_diagnostics(
            model,
            episodes=args.eval_episodes,
            base_seed=10_000 + args.seed * 100,
            action_variant=args.action_variant,
            observation_variant=args.observation_variant,
        )
        summary = summarise_episodes(diagnostics)
        score = score_summary(summary)
        stage_row = {
            "steps_done": steps_done,
            "checkpoint_path": str(checkpoint_base.with_suffix(".zip")),
            "score": score,
            "summary": summary,
        }
        stages.append(stage_row)
        print(json.dumps(stage_row, indent=2))

        if score > best_score:
            best_score = score
            best_path = str(checkpoint_base.with_suffix(".zip"))

    env.close()

    report = {
        "reward_variant": args.reward_variant,
        "action_variant": args.action_variant,
        "observation_variant": args.observation_variant,
        "seed": args.seed,
        "total_timesteps": args.total_timesteps,
        "stage_steps": args.stage_steps,
        "eval_episodes": args.eval_episodes,
        "best_score": best_score,
        "best_checkpoint": best_path,
        "stages": stages,
    }
    report_path = run_dir / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved training report to {report_path}")
    if best_path is not None:
        print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
