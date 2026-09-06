from pathlib import Path
from dataclasses import dataclass
from typing import Any

import yaml


@dataclass
class Config:
    project: dict[str, Any]
    data: dict[str, Any]
    model: dict[str, Any]
    training: dict[str, Any]
    output: dict[str, Any]


def load_config(config_path: str | Path | None = None) -> Config:
    """
    Load the project configuration from configs/config.yaml.
    """

    if config_path is None:
        config_path = (
            Path(__file__).resolve().parents[3]
            / "configs"
            / "config.yaml"
        )

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config_data = yaml.safe_load(file)

    if not isinstance(config_data, dict):
        raise ValueError("Configuration file must contain a YAML mapping.")

    required_sections = [
        "project",
        "data",
        "model",
        "training",
        "output",
    ]

    for section in required_sections:
        if section not in config_data:
            raise ValueError(
                f"Missing required configuration section: {section}"
            )

    return Config(
        project=config_data["project"],
        data=config_data["data"],
        model=config_data["model"],
        training=config_data["training"],
        output=config_data["output"],
    )