from __future__ import annotations

import json
from typing import Any


def safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default)


def try_parse_json_text(text: str) -> Any:
    text = text.strip()
    if not text:
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None
