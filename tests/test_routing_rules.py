import yaml


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_pipeline_order_is_fixed():
    data = load_yaml("routing_rules.yaml")
    assert data["pipeline"] == [
        "bm25_search",
        "vector_search",
        "document_fetch",
        "rerank",
        "answer",
    ]


def test_retry_limit_is_one():
    data = load_yaml("routing_rules.yaml")
    assert data["defaults"]["max_retries"] == 1


def test_hybrid_rule_exists():
    data = load_yaml("routing_rules.yaml")
    rule_names = [rule["name"] for rule in data["rules"]]
    assert "always_use_hybrid" in rule_names


def test_grounding_constraints_are_enabled():
    data = load_yaml("routing_rules.yaml")
    constraints = data["constraints"]
    assert constraints["must_use_context_only"] is True
    assert constraints["must_cite_sources"] is True
    assert constraints["no_hallucination"] is True
    assert constraints["stop_if_no_evidence"] is True
