from pathlib import Path


REQUIRED_ROOT_FILES = [
    "AGENTS.md",
    "mcp_catalog.yaml",
    "routing_rules.yaml",
    "README.md",
    "requirements.txt",
]

REQUIRED_SRC_FILES = [
    "src/__init__.py",
    "src/config_loader.py",
    "src/models.py",
    "src/retrieval.py",
    "src/answerer.py",
    "src/orchestrator.py",
]

REQUIRED_TEST_FILES = [
    "tests/__init__.py",
    "tests/test_structure.py",
    "tests/test_config_loader.py",
    "tests/test_routing_rules.py",
    "tests/test_orchestrator.py",
]


def test_required_root_files_exist():
    for file_path in REQUIRED_ROOT_FILES:
        assert Path(file_path).exists(), f"Missing required root file: {file_path}"


def test_required_src_files_exist():
    for file_path in REQUIRED_SRC_FILES:
        assert Path(file_path).exists(), f"Missing required src file: {file_path}"


def test_required_test_files_exist():
    for file_path in REQUIRED_TEST_FILES:
        assert Path(file_path).exists(), f"Missing required test file: {file_path}"
