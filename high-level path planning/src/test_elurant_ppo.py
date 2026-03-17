"""
Visual and numerical evaluation of the hybrid PPO agent.
Run: python src/test_elurant_ppo.py
"""

import gymnasium as gym
import highway_env  # noqa: F401
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from elurant_ppo import LaneThrottleHybridWrapper, make_config


def main():
    base_env = gym.make("highway-v0", render_mode="human", config=make_config())
    env = LaneThrottleHybridWrapper(base_env)

    model = PPO.load("models/elurant_ppo", env=env)

    print("Running 5 visual episodes...\n")
    for ep in range(5):
        obs, info = env.reset(seed=100 + ep)
        done = truncated = False
        total_reward = 0.0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            env.render()

        print(f"Episode {ep + 1}: reward = {total_reward:.2f}")

    print("\nRunning numerical evaluation (10 episodes)...")
    eval_env = Monitor(LaneThrottleHybridWrapper(gym.make("highway-v0", render_mode="rgb_array", config=make_config())))
    mean, std = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"Mean reward: {mean:.2f} +- {std:.2f}")

    eval_env.close()
    env.close()


if __name__ == "__main__":
    main()
