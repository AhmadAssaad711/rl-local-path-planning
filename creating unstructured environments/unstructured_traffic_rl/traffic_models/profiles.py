"""Programmatic driver profile generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np

ARCHETYPE_NAMES = (
    "conservative",
    "normal",
    "aggressive",
    "erratic",
    "defensive",
    "opportunistic",
)


@dataclass(frozen=True, slots=True)
class DriverProfile:
    """Behavior profile sampled from a continuous parameter space."""

    profile_id: str
    archetype: str
    reaction_time: float
    aggressiveness: float
    risk_tolerance: float
    desired_speed: float
    lane_change_probability: float
    following_distance: float
    overtake_probability: float
    braking_intensity: float
    pothole_avoidance: float
    pedestrian_respect: float
    vehicle_type: str


@dataclass(frozen=True, slots=True)
class ArchetypeSpec:
    reaction_time: tuple[float, float]
    aggressiveness: tuple[float, float]
    risk_tolerance: tuple[float, float]
    desired_speed: tuple[float, float]
    lane_change_probability: tuple[float, float]
    following_distance: tuple[float, float]
    overtake_probability: tuple[float, float]
    braking_intensity: tuple[float, float]
    pothole_avoidance: tuple[float, float]
    pedestrian_respect: tuple[float, float]
    vehicle_types: Sequence[str]


ARCHETYPE_SPECS: Dict[str, ArchetypeSpec] = {
    "conservative": ArchetypeSpec(
        reaction_time=(1.0, 1.8),
        aggressiveness=(0.05, 0.30),
        risk_tolerance=(0.05, 0.25),
        desired_speed=(14.0, 22.0),
        lane_change_probability=(0.02, 0.20),
        following_distance=(10.0, 18.0),
        overtake_probability=(0.05, 0.20),
        braking_intensity=(0.65, 1.00),
        pothole_avoidance=(0.75, 1.00),
        pedestrian_respect=(0.80, 1.00),
        vehicle_types=("car", "taxi", "van"),
    ),
    "normal": ArchetypeSpec(
        reaction_time=(0.6, 1.2),
        aggressiveness=(0.35, 0.60),
        risk_tolerance=(0.30, 0.55),
        desired_speed=(18.0, 27.0),
        lane_change_probability=(0.10, 0.35),
        following_distance=(7.0, 12.0),
        overtake_probability=(0.20, 0.45),
        braking_intensity=(0.45, 0.80),
        pothole_avoidance=(0.45, 0.80),
        pedestrian_respect=(0.50, 0.85),
        vehicle_types=("car", "taxi", "van"),
    ),
    "aggressive": ArchetypeSpec(
        reaction_time=(0.3, 0.8),
        aggressiveness=(0.75, 1.00),
        risk_tolerance=(0.70, 1.00),
        desired_speed=(24.0, 35.0),
        lane_change_probability=(0.45, 0.85),
        following_distance=(3.5, 8.0),
        overtake_probability=(0.60, 1.00),
        braking_intensity=(0.25, 0.60),
        pothole_avoidance=(0.05, 0.45),
        pedestrian_respect=(0.05, 0.35),
        vehicle_types=("car", "motorcycle", "taxi"),
    ),
    "erratic": ArchetypeSpec(
        reaction_time=(0.2, 1.5),
        aggressiveness=(0.15, 1.00),
        risk_tolerance=(0.10, 1.00),
        desired_speed=(15.0, 34.0),
        lane_change_probability=(0.05, 0.95),
        following_distance=(3.0, 16.0),
        overtake_probability=(0.05, 0.95),
        braking_intensity=(0.20, 1.00),
        pothole_avoidance=(0.05, 0.95),
        pedestrian_respect=(0.05, 0.95),
        vehicle_types=("car", "motorcycle", "taxi", "van"),
    ),
    "defensive": ArchetypeSpec(
        reaction_time=(0.8, 1.6),
        aggressiveness=(0.10, 0.40),
        risk_tolerance=(0.05, 0.30),
        desired_speed=(16.0, 24.0),
        lane_change_probability=(0.05, 0.25),
        following_distance=(9.0, 15.0),
        overtake_probability=(0.05, 0.25),
        braking_intensity=(0.70, 1.00),
        pothole_avoidance=(0.65, 1.00),
        pedestrian_respect=(0.75, 1.00),
        vehicle_types=("car", "van", "truck"),
    ),
    "opportunistic": ArchetypeSpec(
        reaction_time=(0.5, 1.0),
        aggressiveness=(0.45, 0.85),
        risk_tolerance=(0.45, 0.85),
        desired_speed=(20.0, 30.0),
        lane_change_probability=(0.20, 0.65),
        following_distance=(5.0, 10.0),
        overtake_probability=(0.30, 0.85),
        braking_intensity=(0.35, 0.75),
        pothole_avoidance=(0.30, 0.85),
        pedestrian_respect=(0.30, 0.75),
        vehicle_types=("car", "motorcycle", "taxi", "van"),
    ),
}


class DriverModelLibrary:
    """Container for a large generated driver profile set."""

    def __init__(self, profiles: Iterable[DriverProfile]):
        self._profiles = tuple(profiles)
        self._profiles_by_id = {profile.profile_id: profile for profile in self._profiles}
        self._by_archetype: dict[str, list[DriverProfile]] = {name: [] for name in ARCHETYPE_NAMES}
        for profile in self._profiles:
            self._by_archetype.setdefault(profile.archetype, []).append(profile)

    @property
    def profiles(self) -> tuple[DriverProfile, ...]:
        return self._profiles

    @property
    def count(self) -> int:
        return len(self._profiles)

    def sample(
        self,
        rng: np.random.Generator,
        archetype_mix: Mapping[str, float] | None = None,
    ) -> DriverProfile:
        """Sample a profile according to an optional archetype mixture."""
        if not archetype_mix:
            return self._profiles[int(rng.integers(0, len(self._profiles)))]
        archetypes = list(archetype_mix.keys())
        weights = np.array(list(archetype_mix.values()), dtype=np.float64)
        weights = weights / weights.sum()
        chosen = str(rng.choice(archetypes, p=weights))
        bucket = self._by_archetype.get(chosen) or list(self._profiles)
        return bucket[int(rng.integers(0, len(bucket)))]

    def get(self, profile_id: str) -> DriverProfile:
        return self._profiles_by_id[profile_id]

    @classmethod
    def generate(
        cls,
        *,
        seed: int = 13,
        profiles_per_archetype: int = 20,
    ) -> "DriverModelLibrary":
        """Generate a reproducible library with 100+ unique profiles."""
        rng = np.random.default_rng(seed)
        profiles: list[DriverProfile] = []
        for archetype in ARCHETYPE_NAMES:
            spec = ARCHETYPE_SPECS[archetype]
            for idx in range(profiles_per_archetype):
                profiles.append(_generate_profile(rng, archetype, spec, idx))
        return cls(profiles)


def _bounded_sample(rng: np.random.Generator, bounds: tuple[float, float], bias: float) -> float:
    low, high = bounds
    raw = np.clip(rng.normal(loc=bias, scale=0.22), 0.0, 1.0)
    return float(low + raw * (high - low))


def _generate_profile(
    rng: np.random.Generator,
    archetype: str,
    spec: ArchetypeSpec,
    idx: int,
) -> DriverProfile:
    bias = (idx + 0.5) / 20.0
    vehicle_type = str(rng.choice(spec.vehicle_types))
    return DriverProfile(
        profile_id=f"{archetype}-{idx:03d}",
        archetype=archetype,
        reaction_time=_bounded_sample(rng, spec.reaction_time, bias),
        aggressiveness=_bounded_sample(rng, spec.aggressiveness, bias),
        risk_tolerance=_bounded_sample(rng, spec.risk_tolerance, 1.0 - bias),
        desired_speed=_bounded_sample(rng, spec.desired_speed, bias),
        lane_change_probability=_bounded_sample(rng, spec.lane_change_probability, bias),
        following_distance=_bounded_sample(rng, spec.following_distance, 1.0 - bias),
        overtake_probability=_bounded_sample(rng, spec.overtake_probability, bias),
        braking_intensity=_bounded_sample(rng, spec.braking_intensity, 1.0 - bias),
        pothole_avoidance=_bounded_sample(rng, spec.pothole_avoidance, 1.0 - bias),
        pedestrian_respect=_bounded_sample(rng, spec.pedestrian_respect, 1.0 - bias),
        vehicle_type=vehicle_type,
    )


def aggressiveness_to_color(aggressiveness: float) -> tuple[int, int, int]:
    """Color mapping used by the pygame renderer."""
    if aggressiveness < 0.33:
        return (80, 210, 110)
    if aggressiveness < 0.66:
        return (240, 210, 70)
    return (235, 90, 80)


def mix_with_defaults(mix: Mapping[str, float] | None = None) -> dict[str, float]:
    """Normalise an archetype mixture and fill omitted archetypes with zero."""
    if not mix:
        weight = 1.0 / len(ARCHETYPE_NAMES)
        return {name: weight for name in ARCHETYPE_NAMES}
    merged = {name: float(mix.get(name, 0.0)) for name in ARCHETYPE_NAMES}
    total = sum(merged.values())
    if total <= 0:
        return {name: 1.0 / len(ARCHETYPE_NAMES) for name in ARCHETYPE_NAMES}
    return {name: value / total for name, value in merged.items()}


DEFAULT_DRIVER_LIBRARY = DriverModelLibrary.generate()
