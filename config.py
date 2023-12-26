import toml
from typing_extensions import TypedDict

CONFIG_FILE = "config.toml"


class Config(TypedDict):
    ...


def read_config() -> Config:
    with open(CONFIG_FILE, "r") as f:
        config = toml.loads(f.read())
    return config
