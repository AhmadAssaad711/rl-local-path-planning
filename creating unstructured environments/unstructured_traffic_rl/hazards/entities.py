"""Road hazards layered on top of highway-env."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from highway_env.vehicle.objects import Obstacle, RoadObject


@dataclass(slots=True)
class Pothole:
    position: np.ndarray
    severity: float
    radius: float
    lane_index: tuple | None


class PedestrianObject(RoadObject):
    """Simple crossing pedestrian updated by the hazard manager."""

    LENGTH = 0.9
    WIDTH = 0.6

    def __init__(
        self,
        road,
        position,
        *,
        heading: float = 0.0,
        speed: float = 0.0,
        intent_probability: float = 0.5,
        crossing_velocity: np.ndarray | None = None,
    ):
        super().__init__(road, position, heading=heading, speed=speed)
        self.intent_probability = float(np.clip(intent_probability, 0.0, 1.0))
        self.crossing_velocity = np.array(crossing_velocity if crossing_velocity is not None else [0.0, 0.0], dtype=np.float64)
        self.solid = True
        self.collidable = True


class HazardManager:
    """Manage hazards, vulnerable road users, and weather side effects."""

    def __init__(self):
        self.potholes: list[Pothole] = []
        self.pedestrians: list[PedestrianObject] = []
        self.static_obstacles: list[Obstacle] = []
        self._managed_objects: list[RoadObject] = []
        self._rng = np.random.default_rng()
        self._scenario = None

    def clear(self, base_env) -> None:
        road = base_env.unwrapped.road
        if road is None:
            return
        for obj in self._managed_objects:
            if obj in road.objects:
                road.objects.remove(obj)
        self._managed_objects.clear()
        self.potholes.clear()
        self.pedestrians.clear()
        self.static_obstacles.clear()

    def reset(self, base_env, scenario, rng: np.random.Generator) -> None:
        self.clear(base_env)
        self._rng = rng
        self._scenario = scenario

        road = base_env.unwrapped.road
        road.weather_friction = scenario.friction
        road.visibility_scale = scenario.visibility

        self._spawn_static_obstacles(base_env)
        self._spawn_potholes(base_env)
        self._spawn_pedestrians(base_env)

    def step(self, base_env, dt: float) -> None:
        if self._scenario is None:
            return
        ego = base_env.unwrapped.vehicle
        road = base_env.unwrapped.road

        for ped in list(self.pedestrians):
            ped.position = ped.position + ped.crossing_velocity * dt
            ped.heading = float(np.arctan2(ped.crossing_velocity[1], ped.crossing_velocity[0] + 1e-6))
            ped.lane_index = road.network.get_closest_lane_index(ped.position, ped.heading)
            ped.lane = road.network.get_lane(ped.lane_index)
            if np.linalg.norm(ped.position - ego.position) > 140:
                if ped in road.objects:
                    road.objects.remove(ped)
                self.pedestrians.remove(ped)
                self._managed_objects.remove(ped)

        if "dynamic_obstacle" in self._scenario.tags and self._rng.random() < self._scenario.obstacle_density * 0.02:
            self._spawn_one_obstacle(base_env, longitudinal=18 + self._rng.uniform(0, 25))

        # Keep pothole and obstacle lists bounded.
        self.potholes = [p for p in self.potholes if np.linalg.norm(p.position - ego.position) < 180]
        self.static_obstacles = [o for o in self.static_obstacles if np.linalg.norm(o.position - ego.position) < 180]

    def metrics(self, base_env) -> dict[str, float]:
        ego = base_env.unwrapped.vehicle
        road = base_env.unwrapped.road

        same_lane_ttc = 10.0
        front_vehicle, _ = road.neighbour_vehicles(ego, ego.lane_index)
        if front_vehicle is not None:
            gap = max(0.1, ego.lane_distance_to(front_vehicle))
            rel_speed = max(0.0, ego.speed - front_vehicle.speed)
            same_lane_ttc = min(10.0, gap / max(rel_speed, 0.1))

        obstacle_distance, obstacle_severity = self._nearest_object_metrics(
            ego,
            self.static_obstacles,
            severity_fn=lambda _: 1.0,
        )
        pothole_distance, pothole_severity = self._nearest_pothole_metrics(ego)
        ped_distance, ped_intent = self._nearest_object_metrics(
            ego,
            self.pedestrians,
            severity_fn=lambda ped: ped.intent_probability,
        )

        local_density = min(1.0, len(road.close_vehicles_to(ego, 90, count=20, see_behind=True, sort=True)) / 20.0)
        scene_risk = float(np.clip((1.0 - self._scenario.visibility) + (1.0 - self._scenario.friction) + self._scenario.hazard_density, 0.0, 3.0) / 3.0)

        return {
            "same_lane_ttc": same_lane_ttc,
            "obstacle_distance": obstacle_distance,
            "obstacle_severity": obstacle_severity,
            "pothole_distance": pothole_distance,
            "pothole_severity": pothole_severity,
            "pedestrian_distance": ped_distance,
            "pedestrian_intent": ped_intent,
            "local_density": local_density,
            "visibility": self._scenario.visibility,
            "friction": self._scenario.friction,
            "scene_risk": scene_risk,
        }

    def lane_clearance_scores(self, base_env) -> dict[int, float]:
        ego = base_env.unwrapped.vehicle
        road = base_env.unwrapped.road
        candidate_lanes = {ego.lane_index}
        candidate_lanes.update(road.network.side_lanes(ego.lane_index))

        scores: dict[int, float] = {}
        for lane_index in candidate_lanes:
            lane_id = lane_index[2]
            score = 0.5

            front_vehicle, rear_vehicle = road.neighbour_vehicles(ego, lane_index)
            if front_vehicle is not None:
                gap = max(0.0, ego.lane_distance_to(front_vehicle, road.network.get_lane(lane_index)))
                score += min(gap / 60.0, 1.0)
            else:
                score += 1.0

            if rear_vehicle is not None:
                rear_gap = abs(rear_vehicle.lane_distance_to(ego, road.network.get_lane(lane_index)))
                score -= max(0.0, 1.0 - rear_gap / 35.0)

            for obstacle in self.static_obstacles:
                if obstacle.lane_index == lane_index:
                    gap = ego.lane_distance_to(obstacle, road.network.get_lane(lane_index))
                    if gap > 0:
                        score -= max(0.0, 1.0 - gap / 45.0)
            for ped in self.pedestrians:
                if ped.lane_index == lane_index:
                    gap = ego.lane_distance_to(ped, road.network.get_lane(lane_index))
                    if gap > 0:
                        score -= ped.intent_probability * max(0.0, 1.0 - gap / 35.0)
            for pothole in self.potholes:
                if pothole.lane_index == lane_index:
                    gap = road.network.get_lane(lane_index).local_coordinates(pothole.position)[0]
                    ego_pos = road.network.get_lane(lane_index).local_coordinates(ego.position)[0]
                    dx = gap - ego_pos
                    if dx > 0:
                        score -= pothole.severity * max(0.0, 1.0 - dx / 35.0)

            scores[lane_id] = float(score)
        return scores

    def reward_penalty(self, base_env, selected_action: int) -> float:
        ego = base_env.unwrapped.vehicle
        metrics = self.metrics(base_env)
        penalty = 0.0

        if metrics["same_lane_ttc"] < 3.0:
            penalty -= 0.35 * (3.0 - metrics["same_lane_ttc"])
        if metrics["pedestrian_distance"] < 15.0 and metrics["pedestrian_intent"] > 0.35 and selected_action != 1:
            penalty -= 0.40 * metrics["pedestrian_intent"]
        if metrics["pothole_distance"] < 12.0 and selected_action != 4:
            penalty -= 0.25 * metrics["pothole_severity"]

        for pothole in self.potholes:
            if np.linalg.norm(pothole.position - ego.position) < (1.5 + pothole.radius):
                penalty -= 0.35 * pothole.severity

        return float(penalty)

    def _spawn_static_obstacles(self, base_env) -> None:
        count = int(max(1, round(self._scenario.obstacle_density * 10)))
        for idx in range(count):
            self._spawn_one_obstacle(base_env, longitudinal=25 + idx * (16 + self._rng.uniform(0, 18)))

    def _spawn_one_obstacle(self, base_env, longitudinal: float) -> None:
        ego = base_env.unwrapped.vehicle
        road = base_env.unwrapped.road
        candidate_lanes = [ego.lane_index, *road.network.side_lanes(ego.lane_index)]
        lane_index = candidate_lanes[int(self._rng.integers(0, len(candidate_lanes)))]
        lane = road.network.get_lane(lane_index)
        coords = lane.local_coordinates(ego.position)
        obstacle = Obstacle.make_on_lane(road, lane_index, coords[0] + longitudinal, speed=0.0)
        road.objects.append(obstacle)
        self.static_obstacles.append(obstacle)
        self._managed_objects.append(obstacle)

    def _spawn_potholes(self, base_env) -> None:
        ego = base_env.unwrapped.vehicle
        road = base_env.unwrapped.road
        count = int(max(1, round(self._scenario.pothole_density * 12)))
        candidate_lanes = [ego.lane_index, *road.network.side_lanes(ego.lane_index)]

        for _ in range(count):
            lane_index = candidate_lanes[int(self._rng.integers(0, len(candidate_lanes)))]
            lane = road.network.get_lane(lane_index)
            ego_s = lane.local_coordinates(ego.position)[0]
            longitudinal = ego_s + float(self._rng.uniform(18.0, 140.0))
            lateral = float(self._rng.uniform(-0.3, 0.3))
            position = np.array(lane.position(longitudinal, lateral), dtype=np.float64)
            severity = float(self._rng.uniform(0.2, 1.0))
            radius = float(0.4 + severity * 1.3)
            self.potholes.append(Pothole(position=position, severity=severity, radius=radius, lane_index=lane_index))

    def _spawn_pedestrians(self, base_env) -> None:
        ego = base_env.unwrapped.vehicle
        road = base_env.unwrapped.road
        count = int(max(1, round(self._scenario.pedestrian_frequency * 8)))
        if count <= 0:
            return
        base_lane = road.network.get_lane(ego.lane_index)
        ego_s = base_lane.local_coordinates(ego.position)[0]

        for _ in range(count):
            longitudinal = ego_s + float(self._rng.uniform(12.0, 80.0))
            lane = base_lane
            heading = lane.heading_at(longitudinal)
            centre = np.array(lane.position(longitudinal, 0.0), dtype=np.float64)
            normal = np.array([-np.sin(heading), np.cos(heading)], dtype=np.float64)
            side = -1.0 if self._rng.random() < 0.5 else 1.0
            spawn = centre + normal * side * (lane.width_at(longitudinal) * 0.9)
            crossing_speed = float(self._rng.uniform(0.7, 1.6))
            velocity = -normal * side * crossing_speed
            ped = PedestrianObject(
                road,
                spawn,
                heading=heading,
                intent_probability=float(self._rng.uniform(0.25, 1.0)),
                crossing_velocity=velocity,
            )
            road.objects.append(ped)
            self.pedestrians.append(ped)
            self._managed_objects.append(ped)

    @staticmethod
    def _nearest_object_metrics(
        ego,
        objects: Iterable,
        *,
        severity_fn,
    ) -> tuple[float, float]:
        best_distance = 150.0
        best_severity = 0.0
        for obj in objects:
            distance = float(np.linalg.norm(obj.position - ego.position))
            if distance < best_distance:
                best_distance = distance
                best_severity = float(severity_fn(obj))
        return best_distance, best_severity

    def _nearest_pothole_metrics(self, ego) -> tuple[float, float]:
        best_distance = 150.0
        best_severity = 0.0
        for pothole in self.potholes:
            distance = float(np.linalg.norm(pothole.position - ego.position))
            if distance < best_distance:
                best_distance = distance
                best_severity = pothole.severity
        return best_distance, best_severity
