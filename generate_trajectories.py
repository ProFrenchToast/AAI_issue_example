import argparse
import os
import random
import numpy as np
import torch as th
from typing import Optional, Dict, List

from stable_baselines3.common.vec_env.patch_gym import _patch_env
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from animalai.environment import AnimalAIEnvironment

from Utils import add_default_args

ENABLE_FIX = True


def get_trajectory_args():
    parser = argparse.ArgumentParser(
        description='Generate trajectories from trained agents or random policy in AnimalAI environments',
    )
    add_default_args(parser)
    parser.add_argument("--num_episodes", default=10, type=int, help="Number of episodes to collect")
    parser.add_argument("--max_steps", default=1000, type=int, help="Maximum steps per episode")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for trajectory generation")

    return parser




def generate_fixed_trajectories(
        configuration_file,
        env_path,
        save_path,
        num_episodes=10,
        max_steps=1000,
        aai_seed=2023,
        args=None,
        device=None,
):
    if device is None:
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

    try:
        port = 5005 + random.randint(0, 1000)

        # Create AnimalAI environment (single env for trajectory collection)
        aai_env = AnimalAIEnvironment(
            seed=aai_seed,
            file_name=env_path,
            arenas_configurations=configuration_file,
            play=False,
            base_port=port,
            inference=True,
            useCamera=True,
            resolution=args.AAI_resolution,
            useRayCasts=False,
            no_graphics=args.no_graphics,
            timescale=args.timescale,
            targetFrameRate=args.target_framerate,
            additional_args=["-force-vulkan"],
            #timeout=600,
        )
        env = UnityToGymWrapper(aai_env, uint8_visual=True, allow_multiple_obs=False, flatten_branched=True)
        env = _patch_env(env)

        # setup the fixed actions to take
        NOOP = 0
        RIGHT = 1
        LEFT = 2
        FORWARD = 3
        FORWARD_RIGHT = 4
        FORWARD_LEFT = 5
        BACKWARD = 6
        BACKWARD_RIGHT = 7
        BACKWARD_LEFT = 8

        actions = [
            [LEFT]*5 + [FORWARD]*30 + [RIGHT]*10+ [FORWARD]*30,
            [RIGHT]*5 + [FORWARD]*30 + [LEFT]*10 + [FORWARD]*30,
        ]
        num_arenas = 2

        # Collect trajectories
        done = False
        for episode in range(num_episodes):
            obs = env.reset()
            episode_obs = []
            episode_actions = []
            episode_rewards = []
            episode_dones = []

            episode_return = 0
            step = 0
            done = False

            while not done and step < max_steps and step < len(actions[episode % len(actions)]):
                episode_obs.append(obs)

                action = actions[episode % len(actions)][step]

                obs, reward, done, info = env.step(action)

                episode_actions.append(action)
                episode_rewards.append(reward)
                episode_dones.append(done)

                episode_return += reward
                step += 1

            #print(f"done {done} step {step < max_steps} len {len(actions[episode % len(actions)])}")
            if done and ENABLE_FIX:
                for remaining_arenas in range(num_arenas-1):
                    env.reset()

    finally:
        if 'env' in locals():
            env.close()
            print("Environment closed")




def main():
    parser = get_trajectory_args()
    args = parser.parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)

    generate_fixed_trajectories(
        configuration_file=args.config_path,
        env_path=args.AAI_path,
        save_path=args.save_path,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        aai_seed=args.AAI_seed,
        args=args,
    )


if __name__ == "__main__":
    main()
