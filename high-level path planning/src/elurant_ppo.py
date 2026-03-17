"""
Hybrid PPO training script using highway-env.

This baseline keeps lane decisions discrete at execution time
(LEFT / IDLE / RIGHT) while learning a continuous throttle signal.
It does not modify the original DQN baseline or its artifacts.
"""

import argparse
import os
from typing import Tuple

import gymnasium as gym
import highway_env  # noqa: F401 - registers highway environments
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

MODEL_PATH = "models/elurant_ppo"


class TimestepMonitorCallback(BaseCallback):
    """Print training progress every fixed number of environment timesteps."""

    def __init__(self, every_n_steps: int = 100):
        super().__init__()
        self.every_n_steps = max(1, int(every_n_steps))
        self._next_print = self.every_n_steps

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_print:
            print(f"[train] timesteps={self.num_timesteps}")
            while self._next_print <= self.num_timesteps:
                self._next_print += self.every_n_steps
        return True


class LaneThrottleHybridWrapper(gym.Wrapper):
    """
    Exposes a 2D continuous action for PPO:
      action[0] -> lane signal in [-1, 1], mapped to {LEFT, IDLE, RIGHT}
      action[1] -> throttle command in [-1, 1], applied as target speed delta

    The wrapped highway environment still receives discrete lane actions.
    """

    LANE_LEFT = 0
    LANE_IDLE = 1
    LANE_RIGHT = 2

    def __init__(
        self,
        env: gym.Env,
        min_speed: float = 10.0,
        max_speed: float = 35.0,
        max_speed_delta_per_step: float = 1.25,
    ):
        super().__init__(env)
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = env.observation_space

        self.min_speed = float(min_speed)
        self.max_speed = float(max_speed)
        self.max_speed_delta_per_step = float(max_speed_delta_per_step)

        self.last_lane_action = self.LANE_IDLE
        self.last_throttle = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_lane_action = self.LANE_IDLE
        self.last_throttle = 0.0
        return obs, info

    def step(self, action):
        lane_signal, throttle = self._parse_action(action)

        ego_lane = int(self.env.unwrapped.vehicle.lane_index[2])
        lane_action = self._lane_signal_to_discrete(lane_signal, ego_lane)

        self._apply_continuous_throttle(throttle)
        obs, reward, terminated, truncated, info = self.env.step(lane_action)

        # Slight smoothness regularization to reduce oscillatory throttle.
        reward += -0.01 * abs(throttle - self.last_throttle)

        self.last_lane_action = lane_action
        self.last_throttle = throttle

        info["lane_action"] = lane_action
        info["throttle"] = throttle
        return obs, reward, terminated, truncated, info

    def _parse_action(self, action: np.ndarray) -> Tuple[float, float]:
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.shape[0] < 2:
            raise ValueError(f"Expected hybrid action of shape (2,), got {a.shape}")
        lane_signal = float(np.clip(a[0], -1.0, 1.0))
        throttle = float(np.clip(a[1], -1.0, 1.0))
        return lane_signal, throttle

    def _lane_signal_to_discrete(self, lane_signal: float, ego_lane: int) -> int:
        if lane_signal < -0.33:
            lane_action = self.LANE_LEFT
        elif lane_signal > 0.33:
            lane_action = self.LANE_RIGHT
        else:
            lane_action = self.LANE_IDLE

        # Keep lane actions valid at road boundaries.
        if ego_lane == 0 and lane_action == self.LANE_LEFT:
            return self.LANE_IDLE
        if ego_lane == 2 and lane_action == self.LANE_RIGHT:
            return self.LANE_IDLE
        return lane_action

    def _apply_continuous_throttle(self, throttle: float) -> None:
        vehicle = self.env.unwrapped.vehicle
        current_target = float(getattr(vehicle, "target_speed", vehicle.speed))
        delta = throttle * self.max_speed_delta_per_step
        new_target = float(np.clip(current_target + delta, self.min_speed, self.max_speed))

        if hasattr(vehicle, "target_speed"):
            vehicle.target_speed = new_target
        else:
            vehicle.speed = new_target


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
            "longitudinal": False,
            "lateral": True,
        },
        "lanes_count": 3,
        "vehicles_count": 20,
        "duration": 40,
    }


def make_env(render_mode: str):
    base_env = gym.make("highway-v0", render_mode=render_mode, config=make_config())
    return LaneThrottleHybridWrapper(base_env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the hybrid elurant PPO agent")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=200000,
        help="Total PPO training timesteps (default: 200000)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of post-training evaluation episodes (default: 10)",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=512,
        help="PPO rollout steps per update (default: 512)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="PPO minibatch size (default: 64)",
    )
    parser.add_argument(
        "--monitor-every-steps",
        type=int,
        default=100,
        help="Print training status every N timesteps (default: 100)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size > args.n_steps:
        raise ValueError("--batch-size must be <= --n-steps")

    env = make_env(render_mode="rgb_array")
    env.reset(seed=42)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
    )

    print(f"Starting PPO training for {args.timesteps} timesteps...")
    monitor_cb = TimestepMonitorCallback(every_n_steps=args.monitor_every_steps)
    model.learn(
        total_timesteps=args.timesteps,
        callback=monitor_cb,
        progress_bar=False,
    )

    os.makedirs("models", exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}.zip")

    eval_env = Monitor(make_env(render_mode="rgb_array"))
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=args.eval_episodes,
    )
    print(
        f"Evaluation over {args.eval_episodes} episodes: "
        f"mean reward = {mean_reward:.2f}, std = {std_reward:.2f}"
    )

    eval_env.close()
    env.close()


if __name__ == "__main__":
    main()
