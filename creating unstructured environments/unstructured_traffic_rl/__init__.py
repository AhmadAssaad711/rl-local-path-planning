"""Unstructured traffic research platform built on top of highway-env."""

from gymnasium.envs.registration import register

from .env.core import UnstructuredTrafficEnv
from .scenarios.generator import ScenarioGenerator
from .scenarios.presets import SCENARIO_LIBRARY
from .traffic_models.profiles import DEFAULT_DRIVER_LIBRARY, DriverModelLibrary

__all__ = [
    "DEFAULT_DRIVER_LIBRARY",
    "DriverModelLibrary",
    "SCENARIO_LIBRARY",
    "ScenarioGenerator",
    "UnstructuredTrafficEnv",
]

try:
    register(
        id="UnstructuredTraffic-v0",
        entry_point="unstructured_traffic_rl.env.core:UnstructuredTrafficEnv",
    )
except Exception:
    # Gymnasium raises if the id already exists in the registry.
    pass
