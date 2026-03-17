"""Procedural scenario generator."""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

from ..traffic_models.profiles import mix_with_defaults
from .presets import SCENARIO_LIBRARY, ScenarioBlueprint, ScenarioConfig

DRIVER_CLASS_PATH = "unstructured_traffic_rl.traffic_models.vehicles.DiverseDriverVehicle"


class ScenarioGenerator:
    """Sample concrete scenario instances from the preset library."""

    def __init__(self, presets: Mapping[str, ScenarioBlueprint] | None = None):
        self.presets = dict(presets or SCENARIO_LIBRARY)

    def names(self) -> list[str]:
        return list(self.presets.keys())

    def sample(
        self,
        name: str | None = None,
        *,
        seed: int | None = None,
        overrides: Mapping[str, float | int | str] | None = None,
    ) -> ScenarioConfig:
        rng = np.random.default_rng(seed)
        if name is None:
            name = str(rng.choice(self.names()))
        blueprint = self.presets[name]
        config = _sample_blueprint(blueprint, rng)
        if overrides:
            config.env_config.update(
                {key: value for key, value in overrides.items() if key in config.env_config}
            )
        return config

    def generate_batch(
        self,
        count: int,
        *,
        seed: int | None = None,
        preset_cycle: Iterable[str] | None = None,
    ) -> list[ScenarioConfig]:
        rng = np.random.default_rng(seed)
        names = list(preset_cycle or self.names())
        return [
            self.sample(names[idx % len(names)], seed=int(rng.integers(0, 2**31 - 1)))
            for idx in range(count)
        ]


def _sample_int(rng: np.random.Generator, bounds: tuple[int, int]) -> int:
    return int(rng.integers(bounds[0], bounds[1] + 1))


def _sample_float(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    return float(rng.uniform(bounds[0], bounds[1]))


def _sample_blueprint(blueprint: ScenarioBlueprint, rng: np.random.Generator) -> ScenarioConfig:
    traffic_density = _sample_float(rng, blueprint.traffic_density)
    hazard_density = _sample_float(rng, blueprint.hazard_density)
    pedestrian_frequency = _sample_float(rng, blueprint.pedestrian_frequency)
    pothole_density = _sample_float(rng, blueprint.pothole_density)
    obstacle_density = _sample_float(rng, blueprint.obstacle_density)
    visibility = _sample_float(rng, blueprint.visibility)
    friction = _sample_float(rng, blueprint.friction)
    lane_count = _sample_int(rng, blueprint.lane_count)
    duration = _sample_int(rng, blueprint.duration)

    env_config = _build_env_config(
        blueprint.base_env_id,
        lane_count=lane_count,
        duration=duration,
        traffic_density=traffic_density,
    )
    return ScenarioConfig(
        slug=blueprint.slug,
        name=blueprint.name,
        description=blueprint.description,
        base_env_id=blueprint.base_env_id,
        layout=blueprint.layout,
        env_config=env_config,
        traffic_density=traffic_density,
        aggressiveness_mix=mix_with_defaults(blueprint.aggressiveness_mix),
        hazard_density=hazard_density,
        pedestrian_frequency=pedestrian_frequency,
        pothole_density=pothole_density,
        obstacle_density=obstacle_density,
        visibility=visibility,
        friction=friction,
        lane_count=lane_count,
        duration=duration,
        tags=blueprint.tags,
        metadata=dict(blueprint.metadata),
    )


def _build_env_config(
    base_env_id: str,
    *,
    lane_count: int,
    duration: int,
    traffic_density: float,
) -> dict:
    common = {
        "other_vehicles_type": DRIVER_CLASS_PATH,
        "simulation_frequency": 15,
        "policy_frequency": 5,
        "duration": duration,
        "show_trajectories": False,
        "real_time_rendering": True,
        "render_agent": True,
        "offscreen_rendering": False,
    }

    if base_env_id == "highway-v0":
        return {
            **common,
            "lanes_count": lane_count,
            "vehicles_count": max(8, int(traffic_density * lane_count * 12)),
            "ego_spacing": 2 + (1.0 - traffic_density) * 2.0,
            "initial_lane_id": min(1, lane_count - 1),
            "screen_width": 1100,
            "screen_height": 280,
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 14,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "normalize": False,
                "absolute": False,
                "order": "sorted",
                "see_behind": True,
            },
            "action": {"type": "DiscreteMetaAction"},
            "collision_reward": -4.0,
            "high_speed_reward": 0.6,
            "right_lane_reward": 0.05,
            "lane_change_reward": -0.02,
            "normalize_reward": False,
            "offroad_terminal": True,
        }
    if base_env_id == "merge-v0":
        return {
            **common,
            "screen_width": 1100,
            "screen_height": 280,
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 16,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "normalize": False,
                "absolute": False,
                "order": "sorted",
                "see_behind": True,
            },
            "action": {"type": "DiscreteMetaAction"},
            "collision_reward": -5.0,
            "high_speed_reward": 0.4,
            "right_lane_reward": 0.0,
            "lane_change_reward": -0.03,
            "normalize_reward": False,
        }
    if base_env_id == "roundabout-v0":
        return {
            **common,
            "screen_width": 780,
            "screen_height": 780,
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 18,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "normalize": False,
                "absolute": True,
                "order": "sorted",
            },
            "action": {
                "type": "DiscreteMetaAction",
                "longitudinal": True,
                "lateral": False,
                "target_speeds": [0, 6, 12, 18],
            },
            "collision_reward": -5.0,
            "high_speed_reward": 0.5,
            "right_lane_reward": 0.0,
            "lane_change_reward": 0.0,
            "normalize_reward": False,
        }
    return {
        **common,
        "screen_width": 780,
        "screen_height": 780,
        "initial_vehicle_count": max(8, int(traffic_density * 18)),
        "spawn_probability": float(np.clip(0.20 + 0.60 * traffic_density, 0.1, 0.95)),
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 18,
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "normalize": False,
            "absolute": True,
            "observe_intentions": False,
        },
        "action": {
            "type": "DiscreteMetaAction",
            "longitudinal": True,
            "lateral": False,
            "target_speeds": [0, 4.5, 9, 13],
        },
        "collision_reward": -5.0,
        "high_speed_reward": 0.6,
        "arrived_reward": 0.8,
        "normalize_reward": False,
    }
