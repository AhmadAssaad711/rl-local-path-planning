"""
Training script for PPO agent on CustomHighwayEnv.
Run: python src/train_ppo_highway.py
"""

import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from envs.custom_highway_env import CustomHighwayEnv
from stable_baselines3 import PPO



def make_env():
    return CustomHighwayEnv()


def main():
    log_dir = "logs/ppo_highway/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Vectorized training environment
    env = make_vec_env(make_env, n_envs=4, monitor_dir=log_dir)

    # PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        n_steps=4096,
        batch_size=128,
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log="./tb_logs/"
    )

    # Evaluation environment + callback
    eval_env = Monitor(make_env())
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/ppo_highway/",
        log_path=log_dir,
        eval_freq=10000,
        deterministic=True,
        render=False
    )

    print("Starting training...\n")
    model.learn(total_timesteps=300000, callback=eval_callback, tb_log_name="ppo_highway")

    model.save("models/ppo_highway/ppo_highway")
    print("\nTraining complete. Model saved to models/ppo_highway/ppo_highway.")

    eval_env.close()
    env.close()


if __name__ == "__main__":
    main()