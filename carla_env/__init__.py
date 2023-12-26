from gymnasium import register

from config import read_config

config = read_config()

register(
    id="Carla-v0",
    entry_point="carla_env.env:CarlaEnv",
    max_episode_steps=config["experiment"]["max_time_episode"],
    reward_threshold=300.0,
)
