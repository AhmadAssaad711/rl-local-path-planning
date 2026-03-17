"""Behavior-level action definitions and mapping to highway-env actions."""

from __future__ import annotations

from enum import IntEnum


class BehaviorAction(IntEnum):
    MAINTAIN_SPEED = 0
    SLOW_DOWN = 1
    CHANGE_LANE_LEFT = 2
    CHANGE_LANE_RIGHT = 3
    AVOID_OBSTACLE = 4
    OVERTAKE = 5
    DEFENSIVE_DRIVING = 6


class BehaviorActionMapper:
    """Translate high-level behavior decisions to highway-env discrete actions."""

    def __init__(self, env):
        self.env = env

    def map(self, action: int) -> int:
        base_env = self.env.base_env.unwrapped
        action_indexes = getattr(base_env.action_type, "actions_indexes", {})
        ego_lane = base_env.vehicle.lane_index[2]
        lane_scores = self.env.hazard_manager.lane_clearance_scores(self.env.base_env)

        def pick(name: str, fallback: str = "IDLE") -> int:
            if name in action_indexes:
                return int(action_indexes[name])
            if fallback in action_indexes:
                return int(action_indexes[fallback])
            return 0

        behavior = BehaviorAction(int(action))
        if behavior == BehaviorAction.MAINTAIN_SPEED:
            return pick("IDLE")
        if behavior == BehaviorAction.SLOW_DOWN:
            return pick("SLOWER")
        if behavior == BehaviorAction.CHANGE_LANE_LEFT:
            return pick("LANE_LEFT")
        if behavior == BehaviorAction.CHANGE_LANE_RIGHT:
            return pick("LANE_RIGHT")
        if behavior == BehaviorAction.DEFENSIVE_DRIVING:
            self.env.defensive_mode_steps = 8
            return pick("SLOWER")

        best_lane = ego_lane
        if lane_scores:
            best_lane = max(lane_scores, key=lane_scores.get)
        if behavior == BehaviorAction.AVOID_OBSTACLE:
            if best_lane < ego_lane:
                return pick("LANE_LEFT", fallback="SLOWER")
            if best_lane > ego_lane:
                return pick("LANE_RIGHT", fallback="SLOWER")
            return pick("SLOWER")
        if behavior == BehaviorAction.OVERTAKE:
            if best_lane < ego_lane and lane_scores[best_lane] > lane_scores.get(ego_lane, -9) + 0.15:
                return pick("LANE_LEFT", fallback="FASTER")
            if best_lane > ego_lane and lane_scores[best_lane] > lane_scores.get(ego_lane, -9) + 0.15:
                return pick("LANE_RIGHT", fallback="FASTER")
            return pick("FASTER")
        return pick("IDLE")
