"""Observation builders for the unstructured traffic environment."""

from __future__ import annotations

import numpy as np

from ..traffic_models.vehicles import VEHICLE_TYPE_CODE

MAX_NEIGHBORS = 8
OBSERVATION_SIZE = 80
LAYOUT_CODE = {
    "corridor": 0.15,
    "urban_corridor": 0.25,
    "narrow_corridor": 0.30,
    "merge": 0.40,
    "intersection": 0.60,
    "urban_junction": 0.65,
    "junction": 0.70,
    "occluded_intersection": 0.75,
    "crosswalk": 0.80,
    "construction": 0.85,
    "roundabout": 0.90,
    "urban_chaos": 1.00,
    "wet_corridor": 0.35,
}


def build_observation(env) -> np.ndarray:
    """Assemble a fixed-size vector observation."""
    base_env = env.base_env.unwrapped
    scenario = env.current_scenario
    metrics = env.hazard_manager.metrics(env.base_env)
    ego = base_env.vehicle
    road = base_env.road
    lane_scores = env.hazard_manager.lane_clearance_scores(env.base_env)
    current_lane_score = lane_scores.get(ego.lane_index[2], 0.0)

    ego_features = np.array(
        [
            np.clip(ego.speed / 40.0, -1.0, 1.5),
            np.clip(getattr(ego, "target_speed", ego.speed) / 40.0, -1.0, 1.5),
            np.clip(ego.heading / np.pi, -1.0, 1.0),
            ego.lane_index[2] / max(1, scenario.lane_count - 1),
            np.clip(current_lane_score / 2.5, -1.0, 1.0),
        ],
        dtype=np.float32,
    )

    close_vehicles = road.close_vehicles_to(ego, 90, count=MAX_NEIGHBORS, see_behind=True, sort=True)
    neighbors = np.zeros((MAX_NEIGHBORS, 7), dtype=np.float32)
    for idx, vehicle in enumerate(close_vehicles[:MAX_NEIGHBORS]):
        rel = vehicle.to_dict(origin_vehicle=ego)
        neighbors[idx] = np.array(
            [
                np.clip(rel["x"] / 80.0, -1.5, 1.5),
                np.clip(rel["y"] / 20.0, -1.5, 1.5),
                np.clip(rel["vx"] / 25.0, -1.5, 1.5),
                np.clip(rel["vy"] / 12.0, -1.5, 1.5),
                np.clip(getattr(vehicle, "aggressiveness", 0.5), 0.0, 1.0),
                np.clip(getattr(vehicle, "risk_tolerance", 0.5), 0.0, 1.0),
                np.clip(getattr(vehicle, "vehicle_type_code", VEHICLE_TYPE_CODE["car"]), 0.0, 1.0),
            ],
            dtype=np.float32,
        )

    risk_features = np.array(
        [
            np.clip(metrics["same_lane_ttc"] / 10.0, 0.0, 1.0),
            np.clip(metrics["obstacle_distance"] / 150.0, 0.0, 1.0),
            np.clip(metrics["obstacle_severity"], 0.0, 1.0),
            np.clip(metrics["pothole_distance"] / 150.0, 0.0, 1.0),
            np.clip(metrics["pothole_severity"], 0.0, 1.0),
            np.clip(metrics["pedestrian_distance"] / 150.0, 0.0, 1.0),
            np.clip(metrics["pedestrian_intent"], 0.0, 1.0),
            np.clip(metrics["local_density"], 0.0, 1.0),
            np.clip(metrics["visibility"], 0.0, 1.0),
            np.clip(metrics["friction"], 0.0, 1.0),
            np.clip(metrics["scene_risk"], 0.0, 1.0),
            np.clip(scenario.hazard_density, 0.0, 1.0),
            np.clip(scenario.pedestrian_frequency, 0.0, 1.0),
        ],
        dtype=np.float32,
    )

    map_features = np.array(
        [
            np.clip(scenario.lane_count / 5.0, 0.0, 1.0),
            np.clip(LAYOUT_CODE.get(scenario.layout, 0.5), 0.0, 1.0),
            np.clip(scenario.traffic_density, 0.0, 1.0),
            np.clip(scenario.obstacle_density, 0.0, 1.0),
            np.clip(scenario.pothole_density, 0.0, 1.0),
            np.clip(scenario.duration / 120.0, 0.0, 1.0),
        ],
        dtype=np.float32,
    )

    observation = np.concatenate(
        [
            ego_features,
            neighbors.reshape(-1),
            risk_features,
            map_features,
        ]
    ).astype(np.float32)
    if observation.shape[0] != OBSERVATION_SIZE:
        raise RuntimeError(f"Observation size mismatch: expected {OBSERVATION_SIZE}, got {observation.shape[0]}")
    return observation
