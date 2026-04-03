from __future__ import annotations

import os
import shlex
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


def _split_env_args(name: str) -> list[str]:
    raw = os.environ.get(name, "").strip()
    return shlex.split(raw) if raw else []


def load_catalog(path: str = "mcp_catalog.yaml") -> dict[str, Any]:
    raw = _load_yaml_file(path)
    servers = raw.get("servers", [])
    if not isinstance(servers, list):
        raise ValueError("mcp_catalog.yaml must define 'servers' as a list")

    filesystem_root = os.environ.get("MCP_FILESYSTEM_ROOT", os.environ.get("DOCS_DIR", "./docs"))
    velocirag_db = os.environ.get("VELOCIRAG_DB", "./.rag/velocirag")

    catalog: dict[str, Any] = {}

    for server in servers:
        if not isinstance(server, dict):
            continue

        name = server.get("name")
        if name == "filesystem":
            command = os.environ.get("FILESYSTEM_MCP_COMMAND", "npx")
            args = _split_env_args("FILESYSTEM_MCP_ARGS") or [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                filesystem_root,
            ]
            catalog["filesystem"] = {
                "command": command,
                "args": args,
                "env": {},
            }
        elif name == "document_parser":
            command = os.environ.get("UNSTRUCTURED_MCP_COMMAND", "uns_mcp")
            args = _split_env_args("UNSTRUCTURED_MCP_ARGS")
            env: dict[str, str] = {}
            if os.environ.get("UNSTRUCTURED_API_KEY"):
                env["UNSTRUCTURED_API_KEY"] = os.environ["UNSTRUCTURED_API_KEY"]
            catalog["document_parser"] = {
                "command": command,
                "args": args,
                "env": env,
            }
        elif name == "retrieval":
            command = os.environ.get("RETRIEVAL_MCP_COMMAND", "velocirag")
            args = _split_env_args("RETRIEVAL_MCP_ARGS") or ["mcp"]
            env = {"VELOCIRAG_DB": velocirag_db}
            catalog["retrieval"] = {
                "command": command,
                "args": args,
                "env": env,
            }
        elif name == "llm_generate":
            command = os.environ.get("LLM_GENERATE_MCP_COMMAND", "ollama-mcp-bridge")
            args = _split_env_args("LLM_GENERATE_MCP_ARGS")
            catalog["llm_generate"] = {
                "command": command,
                "args": args,
                "env": {},
            }

    if "filesystem" in catalog:
        catalog["filesystem_root"] = filesystem_root
    if "retrieval" in catalog:
        catalog["docs_dir"] = filesystem_root

    return catalog
