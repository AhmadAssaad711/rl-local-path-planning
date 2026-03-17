"""Custom highway-env vehicle classes with diverse driver behavior."""

from __future__ import annotations

from typing import Optional

import numpy as np
from highway_env.vehicle.behavior import IDMVehicle

from .profiles import DEFAULT_DRIVER_LIBRARY, DriverProfile, aggressiveness_to_color

VEHICLE_TYPE_CODE = {
    "motorcycle": 0.25,
    "car": 0.50,
    "taxi": 0.65,
    "van": 0.80,
    "truck": 1.00,
}


class DiverseDriverVehicle(IDMVehicle):
    """IDM/MOBIL vehicle with per-instance behavior parameters."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.driver_profile: Optional[DriverProfile] = None
        self.profile_id = ""
        self.driver_archetype = "normal"
        self.aggressiveness = 0.5
        self.risk_tolerance = 0.5
        self.pedestrian_respect = 0.5
        self.pothole_avoidance = 0.5
        self.overtake_probability = 0.5
        self.lane_change_probability = 0.5
        self.behavior_noise = 0.0
        self.vehicle_type_name = "car"
        self.vehicle_type_code = VEHICLE_TYPE_CODE[self.vehicle_type_name]

    def randomize_behavior(self) -> None:
        profile = None
        road = getattr(self, "road", None)
        if road is not None:
            library = getattr(road, "driver_library", DEFAULT_DRIVER_LIBRARY)
            mix = getattr(road, "driver_mix", None)
            rng = getattr(road, "np_random", np.random.default_rng())
            if not isinstance(rng, np.random.Generator):
                seed = int(np.random.randint(0, 2**31 - 1))
                rng = np.random.default_rng(seed)
            profile = library.sample(rng, mix)
        if profile is None:
            profile = DEFAULT_DRIVER_LIBRARY.sample(np.random.default_rng(), None)
        self.apply_profile(profile)

    def apply_profile(self, profile: DriverProfile) -> None:
        """Inject a sampled profile into IDM/MOBIL parameters."""
        self.driver_profile = profile
        self.profile_id = profile.profile_id
        self.driver_archetype = profile.archetype
        self.aggressiveness = profile.aggressiveness
        self.risk_tolerance = profile.risk_tolerance
        self.pedestrian_respect = profile.pedestrian_respect
        self.pothole_avoidance = profile.pothole_avoidance
        self.overtake_probability = profile.overtake_probability
        self.lane_change_probability = profile.lane_change_probability
        self.vehicle_type_name = profile.vehicle_type
        self.vehicle_type_code = VEHICLE_TYPE_CODE.get(profile.vehicle_type, 0.5)

        # IDM gap and longitudinal comfort.
        self.TIME_WANTED = max(0.5, profile.reaction_time * (1.2 - 0.4 * profile.aggressiveness))
        self.DISTANCE_WANTED = max(2.5, profile.following_distance)
        self.COMFORT_ACC_MAX = 1.5 + 2.5 * profile.aggressiveness
        self.COMFORT_ACC_MIN = -(2.0 + 5.0 * profile.braking_intensity)
        self.ACC_MAX = 3.0 + 4.0 * profile.braking_intensity
        self.DELTA = 3.5 + 1.8 * profile.risk_tolerance

        # MOBIL lane-change parameters.
        self.POLITENESS = float(np.clip(1.0 - profile.aggressiveness, 0.0, 1.0))
        self.LANE_CHANGE_MIN_ACC_GAIN = float(
            np.clip(0.45 - 0.35 * profile.aggressiveness, 0.02, 0.50)
        )
        self.LANE_CHANGE_MAX_BRAKING_IMPOSED = float(
            np.clip(1.0 + 2.5 * profile.risk_tolerance, 0.8, 4.5)
        )
        self.LANE_CHANGE_DELAY = float(
            np.clip(0.35 + 1.4 * profile.reaction_time, 0.3, 2.8)
        )

        speed_bonus = {
            "motorcycle": 2.0,
            "car": 0.0,
            "taxi": 1.0,
            "van": -1.0,
            "truck": -4.0,
        }.get(profile.vehicle_type, 0.0)
        self.target_speed = profile.desired_speed + speed_bonus
        self.MAX_SPEED = max(self.target_speed + 8.0, 15.0)

        # Basic mixed-vehicle geometry.
        geometry = {
            "motorcycle": (2.2, 0.9),
            "car": (5.0, 2.0),
            "taxi": (5.2, 2.05),
            "van": (6.0, 2.3),
            "truck": (8.2, 2.6),
        }.get(profile.vehicle_type, (5.0, 2.0))
        self.LENGTH, self.WIDTH = geometry
        self.diagonal = float(np.sqrt(self.LENGTH**2 + self.WIDTH**2))
        self.color = aggressiveness_to_color(profile.aggressiveness)

    def change_lane_policy(self) -> None:
        if self.driver_profile is None:
            self.randomize_behavior()
        rng = getattr(self.road, "np_random", np.random.default_rng())
        if isinstance(rng, np.random.Generator):
            if float(rng.random()) > self.lane_change_probability:
                return
        super().change_lane_policy()

    def act(self, action: dict | str = None):
        if self.driver_profile is None:
            self.randomize_behavior()
        super().act(action)

        # Apply weather/friction after IDM/MOBIL has chosen its command.
        friction = float(np.clip(getattr(self.road, "weather_friction", 1.0), 0.3, 1.2))
        self.action["acceleration"] *= friction

        # A small behavior-specific stochasticity keeps traffic heterogeneous.
        rng = getattr(self.road, "np_random", None)
        if isinstance(rng, np.random.Generator):
            steering_noise = (rng.random() - 0.5) * 0.02 * self.aggressiveness
            self.action["steering"] += steering_noise
