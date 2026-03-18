"""Main Gymnasium environment for unstructured traffic research."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
from gymnasium import spaces

from ..hazards.entities import HazardManager
from ..rendering.overlay import PygameOverlayRenderer
from ..scenarios.generator import ScenarioGenerator
from ..scenarios.presets import ScenarioConfig
from ..traffic_models.profiles import DEFAULT_DRIVER_LIBRARY
from ..traffic_models.vehicles import DiverseDriverVehicle
from .actions import BehaviorAction, BehaviorActionMapper
from .observations import OBSERVATION_SIZE, build_observation


class UnstructuredTrafficEnv(gym.Env):
    """Research platform environment built on top of highway-env."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}

    def __init__(
        self,
        scenario: str | ScenarioConfig = "dense_urban_chaos",
        *,
        render_mode: str | None = None,
        seed: int | None = None,
        resample_on_reset: bool = False,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.generator = ScenarioGenerator()
        self.driver_library = DEFAULT_DRIVER_LIBRARY
        self.rng = np.random.default_rng(seed)
        self.resample_on_reset = resample_on_reset

        self.current_scenario = (
            scenario if isinstance(scenario, ScenarioConfig) else self.generator.sample(str(scenario), seed=seed)
        )
        self.base_env = None
        self.hazard_manager = HazardManager()
        self.overlay_renderer = PygameOverlayRenderer(self)
        self.action_space = spaces.Discrete(len(BehaviorAction))
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(OBSERVATION_SIZE,), dtype=np.float32)
        self.action_mapper: BehaviorActionMapper | None = None
        self.defensive_mode_steps = 0
        self._overlay_attached = False
        self._build_base_env()

    def sample_scenario(
        self,
        name: str | None = None,
        *,
        seed: int | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> ScenarioConfig:
        self.current_scenario = self.generator.sample(name, seed=seed, overrides=overrides)
        return self.current_scenario

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if options and options.get("scenario"):
            scenario_opt = options["scenario"]
            self.current_scenario = (
                scenario_opt
                if isinstance(scenario_opt, ScenarioConfig)
                else self.generator.sample(str(scenario_opt), seed=seed)
            )
            self._build_base_env()
        elif self.resample_on_reset:
            self.current_scenario = self.generator.sample(self.current_scenario.slug, seed=seed)
            self._build_base_env()

        base_obs, info = self.base_env.reset(seed=seed)
        del base_obs
        self._ensure_viewer_safe_actions()
        self._sync_runtime_state()
        self._assign_driver_profiles(force=True)
        self.hazard_manager.reset(self.base_env, self.current_scenario, self.rng)
        self._sync_runtime_state()
        self.action_mapper = BehaviorActionMapper(self)
        self.defensive_mode_steps = 0
        self._overlay_attached = False

        obs = build_observation(self)
        info.update(self._build_info())
        return obs, info

    def step(self, action: int):
        if self.action_mapper is None:
            self.action_mapper = BehaviorActionMapper(self)
        self._ensure_viewer_safe_actions()
        self._apply_ego_behavior(BehaviorAction(int(action)))

        base_action = self.action_mapper.map(action)
        _, base_reward, terminated, truncated, info = self.base_env.step(base_action)

        self._sync_runtime_state()
        self._assign_driver_profiles(force=False)
        dt = 1.0 / float(self.base_env.unwrapped.config.get("policy_frequency", 5))
        self.hazard_manager.step(self.base_env, dt)
        reward = float(base_reward + self._custom_reward(int(action)))

        obs = build_observation(self)
        info.update(self._build_info())
        return obs, reward, terminated, truncated, info

    def render(self):
        image = self.base_env.render()
        viewer = self.base_env.unwrapped.viewer
        if viewer is not None and not self._overlay_attached:
            viewer.set_agent_display(self.overlay_renderer.draw)
            self._overlay_attached = True
            image = self.base_env.render()
        return image

    def close(self):
        self.hazard_manager.clear(self.base_env) if self.base_env is not None else None
        if self.base_env is not None:
            self.base_env.close()
            self.base_env = None

    def _build_base_env(self) -> None:
        if self.base_env is not None:
            self.base_env.close()
        self.base_env = gym.make(
            self.current_scenario.base_env_id,
            config=self.current_scenario.env_config,
            render_mode=self.render_mode,
        )
        self._ensure_viewer_safe_actions()

    def _ensure_viewer_safe_actions(self) -> None:
        action_type = getattr(self.base_env.unwrapped, "action_type", None)
        action_indexes = getattr(action_type, "actions_indexes", None)
        if not isinstance(action_indexes, dict):
            return

        idle_action = action_indexes.get("IDLE")
        if idle_action is None and action_indexes:
            idle_action = next(iter(action_indexes.values()))
        if idle_action is None:
            return

        # highway-env's human viewer always probes lane keys on KEY_UP/KEY_DOWN.
        # Provide no-op aliases for longitudinal-only configs so rendering stays stable.
        action_indexes.setdefault("LANE_LEFT", int(idle_action))
        action_indexes.setdefault("LANE_RIGHT", int(idle_action))

    def _sync_runtime_state(self) -> None:
        road = self.base_env.unwrapped.road
        road.driver_library = self.driver_library
        road.driver_mix = self.current_scenario.aggressiveness_mix
        road.weather_friction = self.current_scenario.friction
        road.visibility_scale = self.current_scenario.visibility

    def _assign_driver_profiles(self, *, force: bool) -> None:
        road = self.base_env.unwrapped.road
        ego = self.base_env.unwrapped.vehicle
        for vehicle in road.vehicles:
            if vehicle is ego or not isinstance(vehicle, DiverseDriverVehicle):
                continue
            if force or not getattr(vehicle, "profile_id", ""):
                profile = self._sample_profile_for_scenario()
                vehicle.apply_profile(profile)

    def _sample_profile_for_scenario(self):
        profile = self.driver_library.sample(self.rng, self.current_scenario.aggressiveness_mix)
        tags = set(self.current_scenario.tags)
        preferred_vehicle_types = set()
        preferred_archetypes = set()

        if "motorcycle" in tags:
            preferred_vehicle_types.add("motorcycle")
            preferred_archetypes.update({"aggressive", "opportunistic"})
        if "taxi" in tags:
            preferred_vehicle_types.add("taxi")
            preferred_archetypes.add("erratic")
        if "mixed_fleet" in tags:
            preferred_vehicle_types.update({"truck", "van", "motorcycle"})
        if "construction" in tags:
            preferred_vehicle_types.update({"truck", "van"})
        if "tailgater" in tags:
            preferred_archetypes.add("aggressive")

        for _ in range(10):
            vehicle_match = not preferred_vehicle_types or profile.vehicle_type in preferred_vehicle_types
            archetype_match = not preferred_archetypes or profile.archetype in preferred_archetypes
            if vehicle_match and archetype_match:
                break
            if vehicle_match and self.rng.random() < 0.35:
                break
            if archetype_match and self.rng.random() < 0.35:
                break
            profile = self.driver_library.sample(self.rng, self.current_scenario.aggressiveness_mix)
        return profile

    def _apply_ego_behavior(self, behavior: BehaviorAction) -> None:
        ego = self.base_env.unwrapped.vehicle
        target_speed = getattr(ego, "target_speed", ego.speed)
        if self.defensive_mode_steps > 0:
            self.defensive_mode_steps -= 1
            target_speed = min(target_speed, ego.speed + 1.0)

        if behavior == BehaviorAction.SLOW_DOWN:
            target_speed -= 2.5
        elif behavior == BehaviorAction.OVERTAKE:
            target_speed += 2.0
        elif behavior == BehaviorAction.AVOID_OBSTACLE:
            target_speed -= 1.5
        elif behavior == BehaviorAction.DEFENSIVE_DRIVING:
            target_speed -= 3.0

        max_speed = getattr(ego, "MAX_SPEED", 40.0)
        ego.target_speed = float(np.clip(target_speed, 0.0, max_speed))

    def _custom_reward(self, action: int) -> float:
        metrics = self.hazard_manager.metrics(self.base_env)
        reward = 0.10
        reward += 0.18 * self.current_scenario.visibility
        reward += 0.12 * self.current_scenario.friction
        reward += 0.14 * min(metrics["same_lane_ttc"], 4.0) / 4.0
        reward -= 0.12 * metrics["scene_risk"]
        reward -= 0.10 * metrics["local_density"]
        reward += self.hazard_manager.reward_penalty(self.base_env, action)

        if action == BehaviorAction.AVOID_OBSTACLE and (
            metrics["obstacle_distance"] < 25.0 or metrics["pothole_distance"] < 20.0
        ):
            reward += 0.10
        if action == BehaviorAction.DEFENSIVE_DRIVING and metrics["scene_risk"] > 0.45:
            reward += 0.08
        if metrics["pedestrian_distance"] < 18.0 and metrics["pedestrian_intent"] > 0.5:
            reward -= 0.30 * metrics["pedestrian_intent"]

        return float(reward)

    def _build_info(self) -> dict[str, Any]:
        metrics = self.hazard_manager.metrics(self.base_env)
        return {
            "scenario": self.current_scenario.slug,
            "scenario_name": self.current_scenario.name,
            "layout": self.current_scenario.layout,
            "driver_model_count": self.driver_library.count,
            "managed_obstacles": len(self.hazard_manager.static_obstacles),
            "managed_pedestrians": len(self.hazard_manager.pedestrians),
            "managed_potholes": len(self.hazard_manager.potholes),
            **metrics,
        }
