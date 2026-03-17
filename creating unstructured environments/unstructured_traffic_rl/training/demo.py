"""Interactive demo and rollout utilities."""

from __future__ import annotations

import argparse
import time

import numpy as np

from ..env.actions import BehaviorAction
from ..env.core import UnstructuredTrafficEnv


def reactive_policy(info: dict) -> int:
    """Simple heuristic policy for quick qualitative demonstrations."""
    if info["pedestrian_distance"] < 16.0 and info["pedestrian_intent"] > 0.45:
        return int(BehaviorAction.DEFENSIVE_DRIVING)
    if info["same_lane_ttc"] < 2.8:
        return int(BehaviorAction.SLOW_DOWN)
    if info["obstacle_distance"] < 18.0 or info["pothole_distance"] < 14.0:
        return int(BehaviorAction.AVOID_OBSTACLE)
    if info["local_density"] < 0.35 and info["same_lane_ttc"] > 4.0:
        return int(BehaviorAction.OVERTAKE)
    return int(BehaviorAction.MAINTAIN_SPEED)


def run_demo(
    scenario: str,
    *,
    episodes: int = 2,
    max_steps: int = 300,
    seed: int = 0,
    render_mode: str | None = "human",
    policy: str = "reactive",
) -> None:
    env = UnstructuredTrafficEnv(scenario, render_mode=render_mode)
    rng = np.random.default_rng(seed)

    try:
        for episode in range(1, episodes + 1):
            _, info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            total_reward = 0.0
            for step in range(max_steps):
                if policy == "random":
                    action = env.action_space.sample()
                else:
                    action = reactive_policy(info)
                _, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if render_mode:
                    env.render()
                    time.sleep(0.05)
                if terminated or truncated:
                    break
            print(
                f"episode={episode} scenario={scenario} reward={total_reward:.2f} "
                f"ttc={info['same_lane_ttc']:.2f} density={info['local_density']:.2f} "
                f"peds={info['managed_pedestrians']} hazards={info['managed_obstacles']}"
            )
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an unstructured traffic simulation demo.")
    parser.add_argument("--scenario", default="dense_urban_chaos")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render-mode", choices=["human", "rgb_array", "none"], default="human")
    parser.add_argument("--policy", choices=["reactive", "random"], default="reactive")
    args = parser.parse_args()

    render_mode = None if args.render_mode == "none" else args.render_mode
    run_demo(
        args.scenario,
        episodes=args.episodes,
        max_steps=args.steps,
        seed=args.seed,
        render_mode=render_mode,
        policy=args.policy,
    )


if __name__ == "__main__":
    main()
