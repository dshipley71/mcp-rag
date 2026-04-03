from pathlib import Path

from src.config import load_catalog
import yaml


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_mcp_catalog_exists_and_loads():
    path = Path("mcp_catalog.yaml")
    assert path.exists(), "mcp_catalog.yaml is missing"
    data = load_yaml(str(path))
    assert isinstance(data, dict), "mcp_catalog.yaml must load as a dictionary"


def test_mcp_catalog_has_required_sections():
    data = load_yaml("mcp_catalog.yaml")
    assert "version" in data
    assert "servers" in data
    assert "global_rules" in data
    assert isinstance(data["servers"], list), "servers must be a list"


def test_routing_rules_exists_and_loads():
    path = Path("routing_rules.yaml")
    assert path.exists(), "routing_rules.yaml is missing"
    data = load_yaml(str(path))
    assert isinstance(data, dict), "routing_rules.yaml must load as a dictionary"


def test_routing_rules_has_required_sections():
    data = load_yaml("routing_rules.yaml")
    assert "version" in data
    assert "defaults" in data
    assert "pipeline" in data
    assert "rules" in data
    assert "constraints" in data
    assert isinstance(data["pipeline"], list), "pipeline must be a list"


def test_load_catalog_maps_real_default_commands(monkeypatch):
    monkeypatch.delenv("RETRIEVAL_MCP_COMMAND", raising=False)
    monkeypatch.delenv("UNSTRUCTURED_MCP_COMMAND", raising=False)
    monkeypatch.delenv("FILESYSTEM_MCP_COMMAND", raising=False)
    monkeypatch.setenv("MCP_FILESYSTEM_ROOT", "/tmp/docs")
    monkeypatch.setenv("VELOCIRAG_DB", "/tmp/velocirag-db")

    catalog = load_catalog("mcp_catalog.yaml")

    assert catalog["retrieval"]["command"] == "velocirag"
    assert catalog["document_parser"]["command"] == "uns_mcp"
    assert catalog["filesystem"]["command"] == "npx"
    assert catalog["filesystem"]["args"][-1] == "/tmp/docs"
    assert catalog["retrieval"]["env"]["VELOCIRAG_DB"] == "/tmp/velocirag-db"

    assert catalog["retrieval"]["args"] == ["mcp"]
