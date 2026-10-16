"""Run in a separate modern Python environment, not ROS Python 3.6."""
import argparse
from pathlib import Path
import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from environment import Task2aCore
from stable_baselines3.common.callbacks import CheckpointCallback

class Task2aEnv(gym.Env):
    def __init__(self, obstacle_count=2):
        super().__init__()
        self.core = Task2aCore()
        self.obstacle_count = obstacle_count
        self.action_space = gym.spaces.Discrete(8)
        self.observation_space = gym.spaces.Box(-1., 1., (2702,), np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return self.core.reset(seed=seed, obstacle_count=self.obstacle_count)

    def step(self, action):
        return self.core.step(action)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--obstacles', type=int, choices=[0, 1, 2], default=0)
    args = parser.parse_args()
    env = Task2aEnv(args.obstacles)
    check_env(env)
    model = DQN('MlpPolicy', env, policy_kwargs={'net_arch': [128, 128]},
                learning_rate=1e-4, buffer_size=20000, learning_starts=2000,
                batch_size=64, gamma=.99, target_update_interval=1000,
                exploration_fraction=.4, exploration_final_eps=.05,
                seed=42, verbose=1, device='cpu')
    checkpoint = CheckpointCallback(
        save_freq=1000,
        save_path=str(Path(__file__).parent / "checkpoints_margin003"),
        name_prefix="dqn",
        save_replay_buffer=False,
        verbose=2,
    )

    model.learn(
        total_timesteps=args.steps,
        callback=checkpoint,
    )
    model.save(str(Path(__file__).parent / 'task2a_dqn'))
    successes = collisions = 0
    for seed in range(10000, 10100):
        obs, _ = env.reset(seed=seed)
        for _ in range(env.core.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                successes += int(info['success'])
                collisions += int(info['collision_or_boundary'])
                break
    print('Held-out placements: success=%d/100 collision_or_boundary=%d/100' %
          (successes, collisions))
