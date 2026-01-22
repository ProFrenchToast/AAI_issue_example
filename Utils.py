import argparse
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean




class RewardLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLoggingCallback, self).__init__(verbose)
        self.episode_rewards = deque(maxlen=10)
        self.episode_reward = 0

    def _on_step(self) -> bool:
        self.episode_reward += self.locals["rewards"][0]
        
        if self.locals["dones"][0]:
            self.episode_rewards.append(self.episode_reward)
            self.episode_reward = 0
        mean_reward = safe_mean(self.episode_rewards)
        self.logger.record("rollout/mean_reward", mean_reward)
        return True

    def get_latest_rewards(self):
        mean_reward = safe_mean(self.episode_rewards)
        return mean_reward


def add_default_args(parser: argparse.ArgumentParser):
    parser.add_argument("--AAI_path", required=True, help="The path to the AnimalAI binary to use. generally something like 'C:\AnimalAI\WINDOWS\Animal-AI.exe' for windows")     # might be able to make this auto find the exe later
    parser.add_argument("--config_path", required=True, help="The path to the config files for the environment as a .yml file.")
    parser.add_argument("--AAI_log", default="./logs", help="The path in which AnimalAI logs are saved")
    parser.add_argument("--tensorboard_log", default="./tensorboardLogs", help="The path in which to save the tensorboard logs")
    parser.add_argument("--AAI_seed", default=2023, type=int, help="the seed used for randomness in AnimalAI")
    parser.add_argument("--AAI_resolution", default=64, type=int, help="The square resolution of the AAI environment camera.")
    parser.add_argument("--no_graphics", default=False, action="store_true", help="set this flag to make animalAI not render graphics to the screen. NOTE: WILL BREAK USE CAMERA OPTION FOR OBSERVATIONS")
    parser.add_argument("--timescale", default=1, type=int, help="multiplier for the deltaTime used by animalAI (this should really be changed in future)")
    parser.add_argument("--target_framerate", default=60, type=int, help="this sets the target framerate for animalAI. set to -1 for uncapped")