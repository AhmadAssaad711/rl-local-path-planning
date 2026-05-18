<div align="center">

# RL Local Path Planning

**Racetrack local path-following with tabular RL and a physics-informed state formulation.**

[Agents](#agent-map) | [Run](#run) | [Layout](#repository-layout)

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![Gymnasium](https://img.shields.io/badge/Gymnasium-highway--env-0B7285?style=flat-square)
![RL](https://img.shields.io/badge/RL-racetrack_path_following-111111?style=flat-square)

</div>

---

## Focus

This repository contains local path-planning and path-following experiments on the `highway-env` racetrack.

The work is organized around one comparison:

1. A basic tabular Q-learning steering policy using lateral error and heading error.
2. A physics-informed tabular Q-learning policy that adds curvature information to the state.
3. A randomized racetrack generalization test comparing both agents.
4. A pure-pursuit steering reference for classical local path following.

## Agent Map

| Component | Path | Role |
| --- | --- | --- |
| Basic RL agent | [`2_basic_rl_agent/basic_rl_agent.py`](2_basic_rl_agent/basic_rl_agent.py) | Q-learning with state `(e_y, e_psi)`. |
| Physics-informed RL agent | [`3_physics_informed_agent/physics_informed_agent.py`](3_physics_informed_agent/physics_informed_agent.py) | Q-learning with path-curvature information in the state. |
| Generalization test | [`generalization_test.py`](generalization_test.py) | Trains both agents and evaluates them on randomized racetrack layouts. |
| Pure pursuit reference | [`classical_baselines/pure_pursuit.py`](classical_baselines/pure_pursuit.py) | Classical lookahead-based local path-following controller. |

## Run

Create a Python environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the basic RL agent:

```powershell
python .\2_basic_rl_agent\basic_rl_agent.py
```

Run the physics-informed RL agent:

```powershell
python .\3_physics_informed_agent\physics_informed_agent.py
```

Run the generalization comparison:

```powershell
python .\generalization_test.py
```

Run the pure-pursuit reference:

```powershell
python .\classical_baselines\pure_pursuit.py
```

Generated plots, Q-tables, logs, and videos are written under `results/` and ignored by Git.

## Repository Layout

```text
.
  2_basic_rl_agent/
    basic_rl_agent.py
  3_physics_informed_agent/
    physics_informed_agent.py
  classical_baselines/
    pure_pursuit.py
  results/
    basic_rl_agent/
    physics_informed_agent/
    generalization/
  generalization_test.py
  requirements.txt
```

## Notes

- The physics-informed agent uses curvature as state information, not as a direct reward term.
- `highway-env` provides the racetrack environment.
- Result artifacts are intentionally excluded so the repository stays code-focused.
