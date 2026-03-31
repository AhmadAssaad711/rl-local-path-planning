"""
TTC-based reward shaping for the Kourani DQN environments.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import gymnasium as gym
import highway_env  # noqa: F401 - register highway-v0
import numpy as np


DEFAULT_TTC_CONFIG: dict[str, Any] = {
    "ttc_safe_threshold": 4.0,
    "ttc_cap": 10.0,
    "ttc_penalty_weight": 0.5,
    "lane_scope": "target",
}


def build_ttc_config(ttc_config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    merged = deepcopy(DEFAULT_TTC_CONFIG)
    if ttc_config:
        merged.update(dict(ttc_config))
    return merged


def _forward_speed(vehicle) -> float:
    return float(vehicle.speed * np.cos(vehicle.heading))


class TTCRewardWrapper(gym.Wrapper):
    """
    Add a time-to-collision penalty on top of the native highway-env reward.
    """

    def __init__(self, env: gym.Env, ttc_config: Mapping[str, Any] | None = None) -> None:
        super().__init__(env)
        self.ttc_config = build_ttc_config(ttc_config)
        self.ttc_safe_threshold = float(max(self.ttc_config["ttc_safe_threshold"], 1e-6))
        self.ttc_cap = float(max(self.ttc_config["ttc_cap"], self.ttc_safe_threshold))
        self.ttc_penalty_weight = float(max(self.ttc_config["ttc_penalty_weight"], 0.0))
        self.lane_scope = str(self.ttc_config["lane_scope"]).lower()
        if self.lane_scope not in {"target", "current"}:
            raise ValueError(
                f"Unsupported lane_scope={self.lane_scope!r}. Expected 'target' or 'current'."
            )

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        ttc_current = self.compute_ttc()
        info = dict(info)
        info["ttc_current"] = float(ttc_current)
        info["ttc_penalty"] = float(self.compute_ttc_penalty(ttc_current))
        return observation, info

    def step(self, action):
        observation, base_reward, terminated, truncated, info = self.env.step(action)
        ttc_current = self.compute_ttc()
        ttc_penalty = self.compute_ttc_penalty(ttc_current)
        shaped_reward = float(base_reward) - float(ttc_penalty)

        info = dict(info)
        info["ttc_current"] = float(ttc_current)
        info["ttc_penalty"] = float(ttc_penalty)
        info["base_reward"] = float(base_reward)
        info["shaped_reward"] = float(shaped_reward)
        return observation, shaped_reward, terminated, truncated, info

    def compute_ttc(self) -> float:
        vehicle = getattr(self.unwrapped, "vehicle", None)
        road = getattr(self.unwrapped, "road", None)
        if vehicle is None or road is None:
            return self.ttc_cap

        lane_index = self._resolve_lane_index(vehicle)
        if lane_index is None:
            return self.ttc_cap

        front_vehicle, _ = road.neighbour_vehicles(vehicle, lane_index)
        if front_vehicle is None:
            return self.ttc_cap

        lane = road.network.get_lane(lane_index)
        ego_s, _ = lane.local_coordinates(vehicle.position)
        front_s, _ = lane.local_coordinates(front_vehicle.position)
        clearance = max(
            0.0,
            float(front_s - ego_s)
            - 0.5
            * float(getattr(vehicle, "LENGTH", 0.0) + getattr(front_vehicle, "LENGTH", 0.0)),
        )

        closing_speed = _forward_speed(vehicle) - _forward_speed(front_vehicle)
        if closing_speed <= 1e-6:
            return self.ttc_cap

        ttc = 0.0 if clearance <= 0.0 else clearance / closing_speed
        return float(np.clip(ttc, 0.0, self.ttc_cap))

    def compute_ttc_penalty(self, ttc_current: float) -> float:
        clipped_ttc = float(np.clip(ttc_current, 0.0, self.ttc_cap))
        normalized_shortfall = max(
            0.0,
            (self.ttc_safe_threshold - clipped_ttc) / self.ttc_safe_threshold,
        )
        return float(self.ttc_penalty_weight * normalized_shortfall)

    def _resolve_lane_index(self, vehicle):
        if self.lane_scope == "current":
            return getattr(vehicle, "lane_index", None)

        target_lane_index = getattr(vehicle, "target_lane_index", None)
        if target_lane_index is not None:
            return target_lane_index
        return getattr(vehicle, "lane_index", None)


def wrap_env_with_ttc(
    env: gym.Env,
    ttc_config: Mapping[str, Any] | None = None,
) -> TTCRewardWrapper:
    return TTCRewardWrapper(env, ttc_config=ttc_config)


def make_ttc_highway_env(
    render_mode: str = "rgb_array",
    config: Mapping[str, Any] | None = None,
    ttc_config: Mapping[str, Any] | None = None,
) -> TTCRewardWrapper:
    base_env = gym.make("highway-v0", render_mode=render_mode, config=dict(config or {}))
    return wrap_env_with_ttc(base_env, ttc_config=ttc_config)
