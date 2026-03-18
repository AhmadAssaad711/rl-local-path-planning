"""Run PPO ablations for the hybrid lane/throttle highway baseline."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import gymnasium as gym
import highway_env  # noqa: F401 - registers highway environments
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO

from elurant_ppo import LaneThrottleHybridWrapper, make_config


VARIANTS = ("baseline", "progress", "progress_clearance", "progress_tactical", "flow_merge")
ACTION_VARIANTS = (
    "deadzone_033",
    "deadzone_010",
    "deadzone_000",
    "intent_guarded",
    "intent_guarded_loose",
    "deadzone_033_calm",
    "intent_guarded_calm",
    "deadzone_033_shielded",
    "intent_guarded_shielded",
)
ACTION_CONFIGS = {
    "deadzone_033": {"lane_change_threshold": 0.33},
    "deadzone_010": {"lane_change_threshold": 0.10},
    "deadzone_000": {"lane_change_threshold": 0.00},
    "deadzone_033_calm": {
        "lane_change_threshold": 0.33,
        "max_speed": 30.0,
        "max_speed_delta_per_step": 0.75,
    },
    "deadzone_033_shielded": {
        "lane_change_threshold": 0.33,
        "throttle_safety_checks": True,
        "min_same_lane_gap": 16.0,
        "min_same_lane_ttc": 2.5,
    },
    "intent_guarded": {
        "lane_action_mode": "intent",
        "lane_intent_decay": 0.70,
        "lane_intent_threshold": 0.35,
        "lane_change_cooldown_steps": 3,
        "lane_safety_checks": True,
        "min_target_front_gap": 14.0,
        "min_target_rear_gap": 10.0,
        "max_target_rear_closing_speed": 7.0,
    },
    "intent_guarded_loose": {
        "lane_action_mode": "intent",
        "lane_intent_decay": 0.75,
        "lane_intent_threshold": 0.25,
        "lane_change_cooldown_steps": 2,
        "lane_safety_checks": True,
        "min_target_front_gap": 10.0,
        "min_target_rear_gap": 8.0,
        "max_target_rear_closing_speed": 9.0,
    },
    "intent_guarded_calm": {
        "lane_action_mode": "intent",
        "lane_intent_decay": 0.72,
        "lane_intent_threshold": 0.35,
        "lane_change_cooldown_steps": 3,
        "lane_safety_checks": True,
        "min_target_front_gap": 16.0,
        "min_target_rear_gap": 12.0,
        "max_target_rear_closing_speed": 6.0,
        "max_speed": 30.0,
        "max_speed_delta_per_step": 0.75,
    },
    "intent_guarded_shielded": {
        "lane_action_mode": "intent",
        "lane_intent_decay": 0.70,
        "lane_intent_threshold": 0.35,
        "lane_change_cooldown_steps": 3,
        "lane_safety_checks": True,
        "min_target_front_gap": 14.0,
        "min_target_rear_gap": 10.0,
        "max_target_rear_closing_speed": 7.0,
        "throttle_safety_checks": True,
        "min_same_lane_gap": 16.0,
        "min_same_lane_ttc": 2.5,
    },
}
OBSERVATION_VARIANTS = ("kinematics", "gap_augmented")


@dataclass
class EpisodeDiagnostics:
    episode_return: float
    steps: int
    crashed: bool
    lane_changes: int
    lane_command_rate: float
    engaged_ratio: float
    mean_speed: float
    min_speed: float
    mean_throttle: float
    stop_ratio: float
    distance_travelled: float


def front_vehicle_state(env: gym.Env) -> tuple[float | None, float | None]:
    vehicle = env.unwrapped.vehicle
    road = env.unwrapped.road
    front, _ = road.neighbour_vehicles(vehicle, vehicle.lane_index)
    if front is None:
        return None, None
    gap = float(max(vehicle.lane_distance_to(front), 0.0))
    return gap, float(front.speed)


def engaged_with_traffic(env: gym.Env, *, radius: float = 40.0) -> bool:
    vehicle = env.unwrapped.vehicle
    nearby = env.unwrapped.road.close_vehicles_to(vehicle, distance=radius, see_behind=True, sort=False)
    return bool(nearby)


def local_flow_speed(env: gym.Env, *, radius: float = 45.0) -> float:
    vehicle = env.unwrapped.vehicle
    nearby = env.unwrapped.road.close_vehicles_to(vehicle, distance=radius, see_behind=True, sort=False)
    if not nearby:
        return float(vehicle.speed)
    return float(np.mean([other.speed for other in nearby]))


class GapFeatureObservationWrapper(gym.ObservationWrapper):
    """Append lane-gap and speed-context features to the kinematics observation."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        base_shape = int(np.prod(env.observation_space.shape))
        self.extra_dim = 10
        self.observation_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(base_shape + self.extra_dim,),
            dtype=np.float32,
        )

    def observation(self, observation):
        base = np.asarray(observation, dtype=np.float32).reshape(-1)
        extras = self._compute_gap_features()
        return np.concatenate([base, extras], dtype=np.float32)

    def _compute_gap_features(self) -> np.ndarray:
        vehicle = self.unwrapped.vehicle
        road = self.unwrapped.road
        lane_count = len(road.network.all_side_lanes(vehicle.lane_index))
        lane_id = int(vehicle.lane_index[2])
        lane_index_norm = -1.0 + 2.0 * lane_id / max(lane_count - 1, 1)
        target_speed = float(getattr(vehicle, "target_speed", vehicle.speed))
        target_speed_norm = np.clip((target_speed - 22.5) / 12.5, -1.0, 1.0)
        current_front_gap, current_front_rel_speed, _, _ = self._lane_metrics(lane_id)
        left_front_gap, _, left_rear_gap, left_rear_closing = self._lane_metrics(lane_id - 1)
        right_front_gap, _, right_rear_gap, right_rear_closing = self._lane_metrics(lane_id + 1)
        return np.array(
            [
                lane_index_norm,
                target_speed_norm,
                current_front_gap,
                current_front_rel_speed,
                left_front_gap,
                left_rear_gap,
                left_rear_closing,
                right_front_gap,
                right_rear_gap,
                right_rear_closing,
            ],
            dtype=np.float32,
        )

    def _lane_metrics(self, lane_id: int) -> tuple[float, float, float, float]:
        vehicle = self.unwrapped.vehicle
        road = self.unwrapped.road
        lane_count = len(road.network.all_side_lanes(vehicle.lane_index))
        if lane_id < 0 or lane_id >= lane_count:
            return -1.0, 0.0, -1.0, 0.0

        lane_index = (vehicle.lane_index[0], vehicle.lane_index[1], lane_id)
        lane = road.network.get_lane(lane_index)
        front, rear = road.neighbour_vehicles(vehicle, lane_index)

        if front is None:
            front_gap_norm = 1.0
            front_rel_speed_norm = 0.0
        else:
            front_gap = float(max(vehicle.lane_distance_to(front, lane), 0.0))
            front_gap_norm = np.clip(front_gap / 60.0, 0.0, 1.0)
            front_rel_speed_norm = np.clip((vehicle.speed - front.speed) / 15.0, -1.0, 1.0)

        if rear is None:
            rear_gap_norm = 1.0
            rear_closing_norm = -1.0
        else:
            rear_gap = float(max(-vehicle.lane_distance_to(rear, lane), 0.0))
            rear_gap_norm = np.clip(rear_gap / 40.0, 0.0, 1.0)
            rear_closing_norm = np.clip((rear.speed - vehicle.speed) / 15.0, -1.0, 1.0)

        return front_gap_norm, front_rel_speed_norm, rear_gap_norm, rear_closing_norm


class RewardAblationWrapper(gym.Wrapper):
    """Replace the stock reward with progress-centric ablations."""

    def __init__(self, env: gym.Env, *, variant: str):
        super().__init__(env)
        self.variant = variant
        self._prev_x = 0.0
        self._prev_lane = 0
        self._prev_throttle = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        vehicle = self.unwrapped.vehicle
        self._prev_x = float(vehicle.position[0])
        self._prev_lane = int(vehicle.lane_index[2])
        self._prev_throttle = 0.0
        return obs, info

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        if self.variant == "baseline":
            reward = float(base_reward)
        else:
            vehicle = self.unwrapped.vehicle
            speed = float(vehicle.speed)
            current_x = float(vehicle.position[0])
            current_lane = int(vehicle.lane_index[2])
            throttle = float(np.asarray(action, dtype=np.float32).reshape(-1)[1])
            delta_x = current_x - self._prev_x
            front_gap, front_speed = self._front_state()
            clear_ahead = front_gap is None or front_gap > 35.0
            tactical = self._tactical_lane_opportunity()
            lane_action = int(info.get("lane_action", LaneThrottleHybridWrapper.LANE_IDLE))
            lane_shift = current_lane - self._prev_lane
            engaged = engaged_with_traffic(self)
            traffic_flow_speed = local_flow_speed(self)

            progress_term = np.clip(delta_x / 25.0, -1.0, 2.0)
            flow_term = np.clip((speed - 20.0) / 10.0, 0.0, 1.0)
            smoothness_penalty = 0.01 * abs(throttle - self._prev_throttle)
            engagement_term = self._engagement_term(front_gap)
            follow_term = self._follow_term(front_gap, front_speed, speed, tactical["available"])

            reward = 0.70 * progress_term + 0.30 * flow_term
            reward -= 2.50 * float(vehicle.crashed)
            reward -= smoothness_penalty

            if self.variant in {"progress_clearance", "progress_tactical"} and clear_ahead:
                reward -= 0.35 * np.clip((20.0 - speed) / 20.0, 0.0, 1.0)
            if self.variant == "progress_tactical" and tactical["available"]:
                reward -= 0.35 * np.clip((22.0 - speed) / 22.0, 0.0, 1.0)
                if lane_action == tactical["best_lane_action"]:
                    reward += 0.10
                if lane_shift == tactical["best_lane_shift"]:
                    reward += 0.10
            if self.variant == "flow_merge":
                reward = 0.55 * progress_term + 0.20 * flow_term + 0.15 * engagement_term + 0.10 * follow_term
                reward -= 2.50 * float(vehicle.crashed)
                reward -= smoothness_penalty
                if clear_ahead:
                    reward -= 0.30 * np.clip((22.0 - speed) / 22.0, 0.0, 1.0)
                if not engaged:
                    reward -= 0.20 * np.clip((traffic_flow_speed - speed) / 10.0, 0.0, 1.0)
                if tactical["available"]:
                    reward -= 0.20 * np.clip((22.0 - speed) / 22.0, 0.0, 1.0)
                    if lane_action == tactical["best_lane_action"]:
                        reward += 0.10
                    if lane_shift == tactical["best_lane_shift"]:
                        reward += 0.10

            info["ablation_progress_term"] = float(progress_term)
            info["ablation_flow_term"] = float(flow_term)
            info["ablation_clear_ahead"] = bool(clear_ahead)
            info["ablation_front_gap"] = None if front_gap is None else float(front_gap)
            info["ablation_tactical_available"] = bool(tactical["available"])
            info["ablation_best_lane_shift"] = int(tactical["best_lane_shift"])
            info["ablation_engagement_term"] = float(engagement_term)
            info["ablation_follow_term"] = float(follow_term)
            info["ablation_engaged"] = bool(engaged)
            info["ablation_local_flow_speed"] = float(traffic_flow_speed)

            self._prev_x = current_x
            self._prev_lane = current_lane
            self._prev_throttle = throttle

        info["reward_variant"] = self.variant
        info["original_reward"] = float(base_reward)
        return obs, float(reward), terminated, truncated, info

    def _front_state(self) -> tuple[float | None, float | None]:
        return front_vehicle_state(self)

    def _engagement_term(self, front_gap: float | None) -> float:
        if front_gap is None:
            return 0.0
        return float(max(0.0, 1.0 - min(abs(front_gap - 24.0) / 24.0, 1.0)))

    def _follow_term(
        self,
        front_gap: float | None,
        front_speed: float | None,
        ego_speed: float,
        tactical_available: bool,
    ) -> float:
        if front_gap is None or front_speed is None or front_gap > 22.0 or tactical_available:
            return 0.0
        return float(max(0.0, 1.0 - min(abs(ego_speed - front_speed) / 10.0, 1.0)))

    def _tactical_lane_opportunity(self) -> dict[str, int | bool]:
        vehicle = self.unwrapped.vehicle
        road = self.unwrapped.road
        lane_count = len(road.network.all_side_lanes(vehicle.lane_index))
        current_lane = int(vehicle.lane_index[2])
        current_front_gap = self._lane_gap(current_lane)

        best_gap = current_front_gap
        best_shift = 0
        best_action = LaneThrottleHybridWrapper.LANE_IDLE
        for shift, action in ((-1, LaneThrottleHybridWrapper.LANE_LEFT), (1, LaneThrottleHybridWrapper.LANE_RIGHT)):
            lane_id = current_lane + shift
            if lane_id < 0 or lane_id >= lane_count:
                continue
            front_gap = self._lane_gap(lane_id)
            rear_gap, rear_closing = self._rear_lane_state(lane_id)
            if front_gap <= current_front_gap + 12.0:
                continue
            if rear_gap < 10.0 and rear_closing > 0.0:
                continue
            if rear_closing > 8.0:
                continue
            if front_gap > best_gap:
                best_gap = front_gap
                best_shift = shift
                best_action = action

        return {
            "available": best_shift != 0,
            "best_lane_shift": best_shift,
            "best_lane_action": best_action,
        }

    def _lane_gap(self, lane_id: int) -> float:
        vehicle = self.unwrapped.vehicle
        road = self.unwrapped.road
        lane_index = (vehicle.lane_index[0], vehicle.lane_index[1], lane_id)
        lane = road.network.get_lane(lane_index)
        front, _ = road.neighbour_vehicles(vehicle, lane_index)
        if front is None:
            return 80.0
        return float(max(vehicle.lane_distance_to(front, lane), 0.0))

    def _rear_lane_state(self, lane_id: int) -> tuple[float, float]:
        vehicle = self.unwrapped.vehicle
        road = self.unwrapped.road
        lane_index = (vehicle.lane_index[0], vehicle.lane_index[1], lane_id)
        lane = road.network.get_lane(lane_index)
        _, rear = road.neighbour_vehicles(vehicle, lane_index)
        if rear is None:
            return 80.0, -5.0
        rear_gap = float(max(-vehicle.lane_distance_to(rear, lane), 0.0))
        rear_closing = float(rear.speed - vehicle.speed)
        return rear_gap, rear_closing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PPO reward ablations for the hybrid highway baseline.")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["baseline", "progress", "progress_clearance"],
        choices=VARIANTS,
        help="Reward variants to train and compare.",
    )
    parser.add_argument(
        "--action-variants",
        nargs="+",
        default=["deadzone_033"],
        choices=ACTION_VARIANTS,
        help="Lane-action mappings to compare.",
    )
    parser.add_argument(
        "--observation-variants",
        nargs="+",
        default=["kinematics"],
        choices=OBSERVATION_VARIANTS,
        help="Observation encodings to compare.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=2048,
        help="Training timesteps per variant and seed.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="Diagnostic episodes per trained model.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0],
        help="Training seeds to run for each variant.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=256,
        help="PPO rollout steps per update.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="PPO minibatch size.",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/ppo_ablation",
        help="Directory where models and summaries will be stored.",
    )
    return parser.parse_args()


def make_env(
    *,
    render_mode: str | None,
    reward_variant: str,
    action_variant: str,
    observation_variant: str,
) -> gym.Env:
    base_env = gym.make("highway-v0", render_mode=render_mode, config=make_config())
    hybrid_env = LaneThrottleHybridWrapper(
        base_env,
        **ACTION_CONFIGS[action_variant],
    )
    wrapped_env: gym.Env = hybrid_env
    if observation_variant == "gap_augmented":
        wrapped_env = GapFeatureObservationWrapper(wrapped_env)
    if reward_variant != "baseline":
        wrapped_env = RewardAblationWrapper(wrapped_env, variant=reward_variant)
    return wrapped_env


def build_model(env: gym.Env, *, seed: int, n_steps: int, batch_size: int) -> PPO:
    return PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        seed=seed,
        verbose=1,
    )


def run_diagnostics(
    model: PPO,
    *,
    episodes: int,
    base_seed: int,
    action_variant: str,
    observation_variant: str,
) -> list[EpisodeDiagnostics]:
    env = make_env(
        render_mode=None,
        reward_variant="baseline",
        action_variant=action_variant,
        observation_variant=observation_variant,
    )
    try:
        diagnostics: list[EpisodeDiagnostics] = []
        for episode_idx in range(episodes):
            obs, info = env.reset(seed=base_seed + episode_idx)
            del info
            done = truncated = False
            episode_return = 0.0
            steps = 0
            lane_changes = 0
            lane_commands = 0
            engaged_steps = 0
            speeds: list[float] = []
            throttles: list[float] = []
            stop_steps = 0
            start_x = float(env.unwrapped.vehicle.position[0])
            prev_lane = int(env.unwrapped.vehicle.lane_index[2])

            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                episode_return += float(reward)
                steps += 1

                vehicle = env.unwrapped.vehicle
                lane = int(vehicle.lane_index[2])
                speed = float(vehicle.speed)
                throttle = float(np.asarray(action, dtype=np.float32).reshape(-1)[1])

                if lane != prev_lane:
                    lane_changes += 1
                prev_lane = lane

                speeds.append(speed)
                throttles.append(throttle)
                if engaged_with_traffic(env):
                    engaged_steps += 1
                if (
                    int(info.get("lane_action", LaneThrottleHybridWrapper.LANE_IDLE))
                    != LaneThrottleHybridWrapper.LANE_IDLE
                ):
                    lane_commands += 1
                if speed < 15.0:
                    stop_steps += 1

            final_vehicle = env.unwrapped.vehicle
            diagnostics.append(
                EpisodeDiagnostics(
                    episode_return=float(episode_return),
                    steps=steps,
                    crashed=bool(final_vehicle.crashed),
                    lane_changes=lane_changes,
                    lane_command_rate=float(lane_commands / max(steps, 1)),
                    engaged_ratio=float(engaged_steps / max(steps, 1)),
                    mean_speed=float(np.mean(speeds)) if speeds else 0.0,
                    min_speed=float(np.min(speeds)) if speeds else 0.0,
                    mean_throttle=float(np.mean(throttles)) if throttles else 0.0,
                    stop_ratio=float(stop_steps / max(steps, 1)),
                    distance_travelled=float(final_vehicle.position[0] - start_x),
                )
            )
        return diagnostics
    finally:
        env.close()


def summarise_episodes(episodes: Iterable[EpisodeDiagnostics]) -> dict[str, float]:
    rows = list(episodes)
    return {
        "mean_episode_return": float(np.mean([row.episode_return for row in rows])),
        "mean_steps": float(np.mean([row.steps for row in rows])),
        "crash_rate": float(np.mean([row.crashed for row in rows])),
        "mean_lane_changes": float(np.mean([row.lane_changes for row in rows])),
        "mean_lane_command_rate": float(np.mean([row.lane_command_rate for row in rows])),
        "mean_engaged_ratio": float(np.mean([row.engaged_ratio for row in rows])),
        "mean_speed": float(np.mean([row.mean_speed for row in rows])),
        "mean_min_speed": float(np.mean([row.min_speed for row in rows])),
        "mean_throttle": float(np.mean([row.mean_throttle for row in rows])),
        "mean_stop_ratio": float(np.mean([row.stop_ratio for row in rows])),
        "mean_distance_travelled": float(np.mean([row.distance_travelled for row in rows])),
    }


def main() -> None:
    args = parse_args()
    if args.batch_size > args.n_steps:
        raise ValueError("--batch-size must be <= --n-steps")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(args.output_dir) / timestamp
    model_dir = root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict[str, object]] = []

    for reward_variant in args.variants:
        for action_variant in args.action_variants:
            for observation_variant in args.observation_variants:
                for seed in args.seeds:
                    print(
                        f"\n=== reward={reward_variant} action={action_variant} "
                        f"obs={observation_variant} seed={seed} timesteps={args.timesteps} ==="
                    )
                    env = make_env(
                        render_mode=None,
                        reward_variant=reward_variant,
                        action_variant=action_variant,
                        observation_variant=observation_variant,
                    )
                    env.reset(seed=seed)
                    model = build_model(env, seed=seed, n_steps=args.n_steps, batch_size=args.batch_size)
                    model.learn(total_timesteps=args.timesteps, progress_bar=False)

                    model_path = model_dir / f"{reward_variant}_{action_variant}_{observation_variant}_seed{seed}"
                    model.save(model_path)

                    diagnostics = run_diagnostics(
                        model,
                        episodes=args.eval_episodes,
                        base_seed=1_000 + 100 * seed,
                        action_variant=action_variant,
                        observation_variant=observation_variant,
                    )
                    summary = summarise_episodes(diagnostics)
                    result = {
                        "reward_variant": reward_variant,
                        "action_variant": action_variant,
                        "observation_variant": observation_variant,
                        "seed": seed,
                        "timesteps": args.timesteps,
                        "eval_episodes": args.eval_episodes,
                        "model_path": str(model_path.with_suffix(".zip")),
                        "summary": summary,
                        "episodes": [asdict(item) for item in diagnostics],
                    }
                    all_results.append(result)

                    env.close()
                    print(
                        json.dumps(
                            {
                                "reward_variant": reward_variant,
                                "action_variant": action_variant,
                                "observation_variant": observation_variant,
                                "seed": seed,
                                **summary,
                            },
                            indent=2,
                        )
                    )

    summary_path = root / "summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nSaved ablation summary to {summary_path}")


if __name__ == "__main__":
    main()
