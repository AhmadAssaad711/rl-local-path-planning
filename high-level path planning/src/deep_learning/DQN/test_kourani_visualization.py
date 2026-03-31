"""
Visual test runner for the trained Kourani DQN with a live policy panel.

The policy panel is drawn through highway-env's viewer extension hook so the
action values appear alongside the highway rendering during a normal rollout.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[4]
PROJECT_VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "logs" / "model_ttc.zip"
MODEL_CANDIDATES = [
    DEFAULT_MODEL_PATH,
    PROJECT_ROOT / "logs" / "model.zip",
    PROJECT_ROOT / "unstructured_scen_codes" / "model_DQN_15.zip",
    Path.cwd() / "logs" / "model_ttc.zip",
    Path.cwd() / "logs" / "model.zip",
]


def maybe_reexec_with_project_venv() -> None:
    if os.environ.get("KOURANI_VISUAL_SKIP_VENV_REEXEC") == "1":
        return

    if not PROJECT_VENV_PYTHON.exists():
        return

    try:
        import highway_env  # noqa: F401
        import matplotlib  # noqa: F401
        import stable_baselines3  # noqa: F401
        return
    except ModuleNotFoundError:
        current_python = Path(sys.executable).resolve()
        venv_python = PROJECT_VENV_PYTHON.resolve()
        if current_python == venv_python:
            raise

        child_env = dict(os.environ)
        child_env["KOURANI_VISUAL_SKIP_VENV_REEXEC"] = "1"
        result = subprocess.run(
            [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]],
            check=False,
            env=child_env,
        )
        raise SystemExit(result.returncode)


maybe_reexec_with_project_venv()

import gymnasium as gym
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import torch
from stable_baselines3 import DQN

from ttc_reward_wrapper import make_ttc_highway_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the trained Kourani DQN with a live state-action value panel."
    )
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stochastic", action="store_true")
    return parser.parse_args()


def make_env() -> gym.Env:
    config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 5,
            "features": ["presence", "x", "y", "vx", "vy"],
            "absolute": False,
        },
        "action": {"type": "DiscreteMetaAction"},
        "lanes_count": 3,
        "vehicles_count": 20,
        "duration": 40,
        "screen_width": 900,
        "screen_height": 220,
    }
    return make_ttc_highway_env(render_mode="human", config=config)


def viewer_closed(env: gym.Env) -> bool:
    return bool(getattr(env.unwrapped, "done", False)) or getattr(env.unwrapped, "viewer", None) is None


def safe_render(env: gym.Env) -> bool:
    try:
        env.render()
    except Exception:
        if viewer_closed(env):
            return False
        raise
    return not viewer_closed(env)


def safe_step(env: gym.Env, action) -> tuple[np.ndarray, float, bool, bool, dict] | None:
    try:
        return env.step(action)
    except Exception:
        if viewer_closed(env):
            return None
        raise


def resolve_model_path(explicit_path: str | Path | None) -> Path:
    if explicit_path:
        candidate = Path(explicit_path).expanduser().resolve()
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Could not find model at {candidate}")

    for candidate in MODEL_CANDIDATES:
        resolved = candidate.expanduser().resolve()
        if resolved.exists():
            return resolved

    raise FileNotFoundError(
        "Could not find a Kourani model automatically. Checked: "
        + ", ".join(str(path.expanduser().resolve()) for path in MODEL_CANDIDATES)
    )


class SB3KouraniAgentAdapter:
    """
    Small adapter that exposes the minimum interface expected by DQNGraphics.
    """

    def __init__(self, model: DQN, env: gym.Env) -> None:
        self.model = model
        self.env = env
        self.device = model.device
        self.config = {"gamma": float(getattr(model, "gamma", 0.8))}
        self.previous_state: np.ndarray | None = None
        self.last_action: int | None = None
        self.action_labels = dict(env.unwrapped.action_type.actions)

    def update(self, state: np.ndarray, action: int | None = None) -> None:
        self.previous_state = np.asarray(state, dtype=np.float32)
        self.last_action = action

    def get_state_action_values(self, state: np.ndarray) -> np.ndarray:
        obs_tensor, _ = self.model.policy.obs_to_tensor(np.asarray(state, dtype=np.float32))
        with torch.no_grad():
            q_values = self.model.policy.q_net(obs_tensor).detach().cpu().numpy()[0]
        return q_values

    def action_distribution(self, state: np.ndarray) -> np.ndarray:
        q_values = self.get_state_action_values(state)
        shifted = q_values - np.max(q_values)
        probabilities = np.exp(shifted)
        probabilities /= np.sum(probabilities)
        return probabilities


class DQNGraphics:
    """
    Graphical visualization of the SB3 DQN state-action values.
    """

    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 80, 80)
    GREEN = (100, 255, 120)

    @classmethod
    def display(cls, agent: SB3KouraniAgentAdapter, surface, sim_surface=None, display_text: bool = True) -> None:
        import pygame

        if agent.previous_state is None:
            return

        q_values = agent.get_state_action_values(agent.previous_state)
        action_distribution = agent.action_distribution(agent.previous_state)
        labels = [agent.action_labels[index] for index in range(len(q_values))]

        width = surface.get_width()
        height = surface.get_height()
        cell_width = max(width // len(q_values), 1)

        pygame.draw.rect(surface, cls.BLACK, (0, 0, width, height), 0)

        q_min = float(np.min(q_values))
        q_max = float(np.max(q_values))
        if np.isclose(q_min, q_max):
            q_min -= 1.0
            q_max += 1.0
        norm = mpl.colors.Normalize(vmin=q_min, vmax=q_max)
        color_map = cm.get_cmap("viridis")

        for action, value in enumerate(q_values):
            color = color_map(norm(float(value)), bytes=True)
            left = cell_width * action
            rect = (left, 0, cell_width, height)
            pygame.draw.rect(surface, color, rect, 0)

            border_color = cls.GREEN if action == int(np.argmax(q_values)) else cls.WHITE
            border_width = 4 if action == agent.last_action else 2
            pygame.draw.rect(surface, border_color, rect, border_width)

            if display_text:
                font = pygame.font.Font(None, 20)
                text_lines = [
                    labels[action],
                    f"Q={value:.2f}",
                    f"p={action_distribution[action]:.2f}",
                ]
                for row_index, line in enumerate(text_lines):
                    text = font.render(line, True, (10, 10, 10), cls.WHITE)
                    surface.blit(text, (left + 10, 12 + row_index * 22))

        footer = pygame.font.Font(None, 22)
        caption = footer.render(
            "Green border = argmax Q, thicker border = executed action",
            True,
            cls.WHITE,
            cls.BLACK,
        )
        surface.blit(caption, (12, height - 28))


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args.model_path)

    env = make_env()
    model = DQN.load(str(model_path), env=env)
    graphics_agent = SB3KouraniAgentAdapter(model=model, env=env)

    try:
        obs, info = env.reset(seed=args.seed)
        graphics_agent.update(obs)

        if not safe_render(env):
            print("Viewer closed before visualization started.")
            return
        env.unwrapped.viewer.set_agent_display(
            lambda surface, sim_surface: DQNGraphics.display(
                graphics_agent, surface, sim_surface, display_text=True
            )
        )
        if not safe_render(env):
            print("Viewer closed before visualization started.")
            return

        for episode in range(args.episodes):
            if episode > 0:
                obs, info = env.reset(seed=args.seed + episode)
                graphics_agent.update(obs)
                if not safe_render(env):
                    print("Viewer closed by user. Exiting visualization.")
                    return

            done = False
            truncated = False
            total_reward = 0.0
            step_count = 0

            while not (done or truncated):
                if viewer_closed(env):
                    print("Viewer closed by user. Exiting visualization.")
                    return
                action, _ = model.predict(obs, deterministic=not args.stochastic)
                action_index = int(np.asarray(action).item())
                graphics_agent.update(obs, action_index)

                step_result = safe_step(env, action)
                if step_result is None:
                    print("Viewer closed by user. Exiting visualization.")
                    return
                obs, reward, done, truncated, info = step_result
                total_reward += float(reward)
                step_count += 1

                if args.max_steps and step_count >= args.max_steps:
                    truncated = True

            print(
                f"Episode {episode + 1}: reward={total_reward:.2f}, "
                f"steps={step_count}, crashed={bool(info.get('crashed', False))}, "
                f"ttc={float(info.get('ttc_current', float('nan'))):.2f}, "
                f"ttc_penalty={float(info.get('ttc_penalty', 0.0)):.3f}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
