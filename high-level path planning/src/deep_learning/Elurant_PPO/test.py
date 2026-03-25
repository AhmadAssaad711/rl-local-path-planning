"""
Visual and numerical evaluation of PPO agent on CustomHighwayEnv.
Run: python src/test_ppo_highway.py
"""

import time
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from envs.custom_highway_env import CustomHighwayEnv


def main():
    model = PPO.load("models/ppo_highway/ppo_highway")

    print("Running 5 visual episodes...\n")
    for ep in range(5):
        print(f"--- Episode {ep + 1} ---")

        env = CustomHighwayEnv(render_mode="human")
        obs, info = env.reset(seed=100 + ep)

        done = truncated = False
        total_reward = 0.0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

            total_reward += reward
            env.render()  # ✅ REQUIRED
            time.sleep(0.05)

        print(f"Episode {ep + 1}: reward = {total_reward:.2f}")
        env.close()

    print("\nRunning numerical evaluation (10 episodes)...")
    eval_env = Monitor(CustomHighwayEnv(render_mode="rgb_array"))
    mean, std = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"Mean reward: {mean:.2f} +- {std:.2f}")

    eval_env.close()


if __name__ == "__main__":
    main()