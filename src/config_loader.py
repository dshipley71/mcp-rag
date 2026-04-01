from pathlib import Path
from typing import Any

import yaml


def load_yaml_file(path: str) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Config file must load as a dictionary: {path}")

    return data


def load_mcp_catalog() -> dict[str, Any]:
    return load_yaml_file("mcp_catalog.yaml")


def load_routing_rules() -> dict[str, Any]:
    return load_yaml_file("routing_rules.yaml")
