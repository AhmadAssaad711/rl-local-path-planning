"""
Baseline DQN training script using Stable-Baselines3 with highway-env.
Follows the formulation by Edouard Leurent in the highway-env examples.
Used as a reference for comparison with our custom CNN-DQN agent.
"""

import os

import gymnasium as gym
import highway_env  # noqa: F401 - registers highway environments
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy


def main():
    config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 5,
            "features": ["presence", "x", "y", "vx", "vy"],
            "absolute": False,
        },
        "action": {
            "type": "DiscreteMetaAction",
        },
        "lanes_count": 3,
        "vehicles_count": 20,
        "duration": 40,
    }

    env = gym.make("highway-v0", render_mode="rgb_array", config=config)
    env.reset()

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=5e-4,
        buffer_size=15000,
        learning_starts=200,
        batch_size=32,
        gamma=0.8,
        target_update_interval=50,
        verbose=1,
    )

    print("Starting training for 200000 timesteps...")
    model.learn(total_timesteps=200000)

    os.makedirs("models", exist_ok=True)
    model.save("models/elurant_dqn")
    print("Model saved to models/elurant_dqn.zip")

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Evaluation over 10 episodes: mean reward = {mean_reward:.2f}, std = {std_reward:.2f}")

    env.close()


if __name__ == "__main__":
    main()
