from typing import Any

from src.config import load_catalog, _load_yaml_file


def load_yaml_file(path: str) -> dict[str, Any]:
    return _load_yaml_file(path)


def load_mcp_catalog() -> dict[str, Any]:
    return load_catalog("mcp_catalog.yaml")


def load_routing_rules() -> dict[str, Any]:
    return load_yaml_file("routing_rules.yaml")
