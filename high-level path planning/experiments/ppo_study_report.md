# PPO Hybrid Control Study

## Scope

Goal: fix the PPO hybrid lane/throttle agent so it no longer collapses into brake-heavy or crash-heavy behavior.

Constraint followed: `src/test_elurant_ppo.py` was not used. All validation was done with custom diagnostic rollouts and saved JSON summaries.

## Code Added / Changed

- `src/elurant_ppo.py`
  - Added configurable lane dead-zone.
  - Added persistent lane-intent mode with cooldown and lane-safety checks.
  - Added optional same-lane throttle safety cap based on front gap / TTC.
- `src/elurant_ppo_ablation.py`
  - Added reward, action, and observation ablation support.
  - Added gap-augmented observations.
  - Added richer diagnostics: crash rate, lane changes, lane command rate, stop ratio, distance.
- `src/elurant_ppo_train_select.py`
  - Added staged training with checkpoint selection using custom diagnostics.

## Experiment Log

| Phase | Artifact | What Was Tested | Key Result | Verdict |
|---|---|---|---|---|
| Reward pilot | `experiments/ppo_ablation/20260317_230503/summary.json` | `baseline` vs `progress` vs `progress_clearance` | Reward shaping reduced unnecessary braking and improved distance, but lane changes stayed at `0.0`. | Reward mattered, but was not enough. |
| Lane-threshold sweep | `experiments/ppo_ablation/20260317_231752/summary.json` | `deadzone_033`, `deadzone_010`, `deadzone_000` | Removing the dead-zone produced lane changes, but crash rate jumped to `0.667`. | Naive dead-zone removal was rejected. |
| Reward + action + obs matrix | `experiments/ppo_ablation/20260317_233529/summary.json` | `progress_clearance` with threshold vs intent modes, `kinematics` vs `gap_augmented` | `gap_augmented` improved flow; `intent_guarded + kinematics` produced lane changes with `0.0` crash over the short pilot; several other combos crashed or regressed. | Useful signal, but not stable enough yet. |
| Longer rerun | `experiments/ppo_ablation/20260317_234350/summary.json` | 2048-step rerun of the best short-pilot configs | Several promising 512-step policies collapsed back into conservative braking or unstable lane usage. | Fixed-horizon longer training alone was rejected. |
| Tactical reward | `experiments/ppo_ablation/20260318_002230/summary.json` | Added explicit tactical-lane opportunity reward | Higher speed, but crash rates between `0.4` and `1.0` depending on action mode. | Rejected as too crash-prone. |
| Shielded variants | `experiments/ppo_ablation/20260318_003346/summary.json` | Added same-lane throttle safety cap | `intent_guarded_shielded + gap_augmented + progress_clearance` reached `crash_rate=0.05`, `mean_lane_changes=0.85`, `mean_speed=21.76`, `distance=832.05` over 20 eval episodes. | First viable configuration. |
| Stage-wise training, seed 0 | `experiments/ppo_final/20260318_004110/seed_0/training_report.json` | Checkpoint selection every 256 steps up to 1024 | Best checkpoint was `step_256`, with `0.0` crash and `876.87 m` distance over 10 eval episodes. | Strong safe-flow model, almost no lane changes. |
| Stage-wise training, seed 1 | `experiments/ppo_final/20260318_004606/seed_1/training_report.json` | Same staged training on a second seed | Best checkpoint was `step_768`, with `0.0` crash, `0.3` mean lane changes, and `869.51 m` distance over 10 eval episodes. | Best balance of flow and occasional lane use. |

## Final Model Selection

Selected model:

- `models/elurant_ppo_best_shielded_gap_augmented.zip`
- Source checkpoint:
  - `experiments/ppo_final/20260318_004606/seed_1/checkpoints/step_768.zip`

Final configuration:

- Reward variant: `progress_clearance`
- Action variant: `intent_guarded_shielded`
- Observation variant: `gap_augmented`
- Seed: `1`
- Checkpoint chosen by staged custom diagnostics, not by final training timestep

## Final Stress Test

Two strongest candidates were stress-tested over 30 episodes:

| Candidate | Crash Rate | Mean Lane Changes | Mean Speed | Stop Ratio | Mean Distance |
|---|---:|---:|---:|---:|---:|
| `seed0_step256` | `0.00` | `0.1333` | `21.4255` | `0.0000` | `858.8813` |
| `seed1_step768` | `0.00` | `0.2667` | `21.5316` | `0.0000` | `863.0391` |

The `seed1_step768` model was chosen because it kept the same zero-crash result while showing more lane changes and slightly better distance.

## What Worked

- Reward shaping away from pure safety/clearance optimization prevented the original brake-to-zero collapse.
- Gap-augmented observations improved throughput and reduced unnecessary slowing.
- Longitudinal safety shielding was the largest safety improvement.
- Stage-wise checkpoint selection was necessary because some models became worse with more PPO updates.

## What Failed

- Pure reward changes without action / safety changes.
- Pure lane-threshold reduction without safety logic.
- Tactical reward bonus without enough safety structure.
- Calm throttle limits on their own.
- Blindly training longer and assuming the latest checkpoint is best.

## Remaining Limitations

- Lane changes are still sparse. The final model does not exhibit aggressive tactical overtaking.
- The best result uses a safety shield, so this is not a pure unconstrained PPO policy.
- The model is tuned for this specific `highway-v0` hybrid setup and should be revalidated if traffic density or observation settings change.
