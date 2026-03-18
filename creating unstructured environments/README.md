# Unstructured Traffic RL

Research platform for autonomous driving behavior studies in chaotic and partially unstructured traffic, built by extending `highway-env` rather than replacing it.

## What is implemented

- `unstructured_traffic_rl/` modular package with `env`, `traffic_models`, `scenarios`, `hazards`, `rendering`, `training`, and `utils` modules.
- `UnstructuredTrafficEnv(gym.Env)` Gymnasium-compatible wrapper on top of `highway-env`.
- 20 configurable scenario presets implemented as generalized configurations, not 20 separate environment classes.
- 120 generated driver profiles across six archetypes: conservative, normal, aggressive, erratic, defensive, opportunistic.
- Real-time pygame overlays for driver aggressiveness, potholes, pedestrians, hazards, and scenario telemetry.
- Procedural scenario generation for batch datasets and RL research.

## Repository structure

```text
creating unstructured environments/
|-- datasets/
|-- tests/
|-- unstructured_traffic_rl/
|   |-- env/
|   |-- hazards/
|   |-- rendering/
|   |-- scenarios/
|   |-- traffic_models/
|   |-- training/
|   `-- utils/
|-- pyproject.toml
`-- README.md
```

Earlier single-scenario experiments remain outside this folder under `../high-level path planning/`. The reusable unstructured-environment platform lives entirely inside this directory.

## Scenario library

1. Dense informal intersection
2. Aggressive merge traffic
3. Chaotic roundabout
4. Pedestrian swarm crossing
5. Motorcycle lane splitting
6. Sudden obstacle emergence
7. Pothole avoidance corridor
8. Informal overtaking traffic
9. Stop-and-go urban congestion
10. Narrow road negotiation
11. Blind intersection with occlusion
12. Erratic taxi behavior
13. Aggressive tailgater scenario
14. Pedestrian hesitation conflict
15. Mixed vehicle types traffic
16. Double parking obstruction
17. Construction zone traffic
18. Rain or low friction environment
19. Multi-vehicle negotiation junction
20. Dense urban chaos scenario

## Quick start

From the `creating unstructured environments/` directory:

```powershell
..\.venv\Scripts\python.exe -m unstructured_traffic_rl.training.demo --scenario dense_urban_chaos --episodes 1 --render-mode human
```

For an offscreen smoke rollout:

```powershell
..\.venv\Scripts\python.exe -m unstructured_traffic_rl.training.demo --scenario pothole_avoidance_corridor --episodes 1 --steps 50 --render-mode none
```

Generate a procedural batch of scenario manifests:

```powershell
..\.venv\Scripts\python.exe -m unstructured_traffic_rl.training.scenario_batch --count 1000 --output datasets/generated_scenarios.json
```

Run smoke tests:

```powershell
..\.venv\Scripts\python.exe -m unittest tests.test_platform_smoke
```

Optional SB3 training example:

```powershell
..\.venv\Scripts\python.exe -m unstructured_traffic_rl.training.sb3_train --scenario dense_urban_chaos --algo ppo --timesteps 20000
```

By default, trained SB3 checkpoints are written to a local `models/` directory inside this folder.

## RL interface

`UnstructuredTrafficEnv` exposes:

- Observation: ego state, neighbor state summaries, hazard metrics, and map context.
- Actions: maintain speed, slow down, lane left, lane right, avoid obstacle, overtake, defensive driving.
- Reward: base `highway-env` reward plus visibility, friction, TTC, hazard, pothole, and pedestrian shaping.

## Notes

- Vehicle colors encode aggressiveness: green calm, yellow normal, red aggressive.
- Potholes render as circles whose size and color represent severity.
- Pedestrian intent is rendered by color intensity.
- The system is designed to scale by procedural scenario generation rather than hand-authored environment subclasses.
