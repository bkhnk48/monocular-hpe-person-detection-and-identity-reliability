from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    """
    Load YAML configuration and validate essential fields.

    Paths in the configuration are interpreted relative to the
    repository root (the parent of the configs directory).
    """
    config_path = Path(config_path).expanduser().resolve()

    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")

    required_sections = [
        "data",
        "gt",
        "detections",
        "joints",
        "validation",
        "association",
        "solvers",
        "evaluation",
        "experiments",
    ]

    for section in required_sections:
        if section not in config:
            raise KeyError(f"Missing configuration section: {section}")

    shared_joints = config["joints"].get("shared", [])
    if not shared_joints:
        raise ValueError("joints.shared must contain at least one joint.")

    for key in ("gt_file", "detections_file", "processed_dir"):
        if key not in config["data"]:
            raise KeyError(f"Missing data.{key} in configuration.")

    # The repository root is one level above configs/.
    project_root = config_path.parent.parent

    config["_project_root"] = project_root
    config["_config_path"] = config_path

    for key in ("gt_file", "detections_file", "processed_dir"):
        path = Path(config["data"][key]).expanduser()
        if not path.is_absolute():
            path = project_root / path
        config["data"][key] = path.resolve()

    return config
