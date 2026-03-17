"""Smoke tests for the unstructured traffic platform."""

from __future__ import annotations

import unittest

from unstructured_traffic_rl import DEFAULT_DRIVER_LIBRARY, SCENARIO_LIBRARY, ScenarioGenerator, UnstructuredTrafficEnv


class PlatformSmokeTests(unittest.TestCase):
    def test_driver_library_has_required_scale(self):
        self.assertGreaterEqual(DEFAULT_DRIVER_LIBRARY.count, 100)

    def test_scenario_library_has_required_count(self):
        self.assertEqual(len(SCENARIO_LIBRARY), 20)

    def test_procedural_generation_scales(self):
        batch = ScenarioGenerator().generate_batch(256, seed=4)
        self.assertEqual(len(batch), 256)
        self.assertGreater(len({item.slug for item in batch}), 5)

    def test_environment_smoke(self):
        for scenario_name in [
            "dense_urban_chaos",
            "aggressive_merge_traffic",
            "chaotic_roundabout",
            "pothole_avoidance_corridor",
        ]:
            env = UnstructuredTrafficEnv(scenario_name, render_mode=None)
            obs, info = env.reset(seed=0)
            self.assertEqual(obs.shape[0], 80)
            self.assertIn("scenario", info)
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            self.assertEqual(obs.shape[0], 80)
            self.assertIsInstance(reward, float)
            self.assertIn("same_lane_ttc", info)
            self.assertIn(terminated, [True, False])
            self.assertIn(truncated, [True, False])
            env.close()


if __name__ == "__main__":
    unittest.main()
