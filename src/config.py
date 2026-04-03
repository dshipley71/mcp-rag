from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def _load_yaml_file(path: str) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Config file must load as a dictionary: {path}")

    return data


def _default_server_config(server_name: str) -> dict[str, Any]:
    filesystem_root = os.environ.get("MCP_FILESYSTEM_ROOT", "./docs")

    defaults: dict[str, dict[str, Any]] = {
        "filesystem": {
            "command": os.environ.get("FILESYSTEM_MCP_COMMAND", "npx"),
            "args": [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                filesystem_root,
            ],
        },
        "document_parser": {
            "command": os.environ.get("UNSTRUCTURED_MCP_COMMAND", "unstructured-mcp"),
            "args": [],
        },
        "retrieval": {
            "command": os.environ.get("RETRIEVAL_MCP_COMMAND", "velocirag-mcp"),
            "args": [],
        },
        "llm_generate": {
            "command": os.environ.get("LLM_GENERATE_MCP_COMMAND", "ollama-mcp-bridge"),
            "args": [],
        },
    }

    if server_name not in defaults:
        raise KeyError(f"Unsupported MCP server role in catalog: {server_name}")

    return defaults[server_name].copy()


def load_catalog(path: str = "mcp_catalog.yaml") -> dict[str, Any]:
    """
    Load the repository MCP catalog and normalize it into an executable runtime catalog.

    Supported inputs:
    1. High-level repo catalog with a `servers` list.
    2. Already-normalized runtime catalog keyed by role name.
    """
    data = _load_yaml_file(path)

    if "servers" not in data:
        return data

    runtime_catalog: dict[str, Any] = {}
    for entry in data.get("servers", []):
        if not isinstance(entry, dict):
            continue

        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            continue

        runtime_catalog[name] = _default_server_config(name)

    filesystem_root = os.environ.get("MCP_FILESYSTEM_ROOT")
    if filesystem_root:
        runtime_catalog["filesystem_root"] = filesystem_root

    runtime_catalog["_raw_catalog"] = data
    return runtime_catalog
