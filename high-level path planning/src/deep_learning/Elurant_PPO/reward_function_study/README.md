# PPO Reward Function Study

This folder runs a sequential PPO sweep over the native `highway-v0` reward terms while keeping the continuous-control environment fixed.

Default study design:

- `collision_reward ∈ {-0.75, -1.0, -1.5}`
- `high_speed_reward ∈ {0.25, 0.4, 0.6}`
- `right_lane_reward ∈ {0.0, 0.05, 0.1}`
- `27` grid runs + `3` extra baseline repeats = `30` total runs

The training script logs reward-study behavior metrics to TensorBoard:

- collision rate
- off-road rate
- mean forward speed
- right-lane occupancy ratio
- throttle variance
- steering variance
- mean absolute throttle change
- mean absolute steering change

## Commands

Train the full sweep sequentially:

```powershell
python ".\high-level path planning\src\deep_learning\Elurant_PPO\reward_function_study\run_reward_study.py" train
```

You can also run the file directly with no subcommand, and it will default to `train`:

```powershell
python ".\high-level path planning\src\deep_learning\Elurant_PPO\reward_function_study\run_reward_study.py"
```

If you want training to also record one video per finished model, opt in explicitly:

```powershell
python ".\high-level path planning\src\deep_learning\Elurant_PPO\reward_function_study\run_reward_study.py" --auto-video
```

Record one episode for every trained model:

```powershell
python ".\high-level path planning\src\deep_learning\Elurant_PPO\reward_function_study\run_reward_study.py" video
```

Open TensorBoard on the study logs:

```powershell
tensorboard --logdir ".\high-level path planning\src\deep_learning\Elurant_PPO\reward_function_study\tb_logs"
```

Study outputs are written to:

- `m/` for saved PPO checkpoints and manifests
- `l/` for monitor and evaluation logs
- `tb/` for TensorBoard
- `v/` for one-episode evaluation recordings
- `s/reward_grid_summary.csv` for the aggregated run table
