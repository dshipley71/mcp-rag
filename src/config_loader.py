from pathlib import Path
import yaml


def load_yaml_file(path: str) -> dict:
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing config file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_mcp_catalog() -> dict:
    return load_yaml_file("mcp_catalog.yaml")


def load_routing_rules() -> dict:
    return load_yaml_file("routing_rules.yaml")
