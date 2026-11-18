"""Canonical structure, key and diff of the config of a JSON task.

The async runner rejects a proposal when its key is the key of an earlier
config, and the proposal prompt gives an earlier config as its changed keys.
Each float becomes its float32 value: 0.0010000000474974513 and 0.001 are the
same value. An int stays exact, thus an int and a float are different values.
A bool is never equal to a number.
"""

import json
from typing import Any, List, Optional, Tuple

import numpy as np

from .languages import get_evolve_comment_prefix

ABSENT_TEXT = "(absent)"
# The value of a key that one side of a diff does not have
_ABSENT = object()


def is_json_language(language: str) -> bool:
    """Return True for the language of a JSON task (json or json5)."""
    return language.strip().lower() in ("json", "json5")


def to_float32(value: Any) -> Any:
    """Replace each float of a parsed JSON value with its float32 value."""
    if isinstance(value, float):
        return float(np.float32(value))
    if isinstance(value, list):
        return [to_float32(item) for item in value]
    if isinstance(value, dict):
        return {key: to_float32(item) for key, item in value.items()}
    return value


def canonical_config(text: str, language: str = "json") -> Any:
    """Parse a config without its comment lines, with float32 floats.

    Raise ValueError when the text does not parse.
    """
    prefix = get_evolve_comment_prefix(language)
    lines = text.splitlines()
    kept = [line for line in lines if not line.lstrip().startswith(prefix)]
    return to_float32(json.loads("\n".join(kept)))


def config_key(text: str, language: str = "json") -> Optional[str]:
    """Return the key of a config, or None when the text does not parse."""
    try:
        return json.dumps(canonical_config(text, language), sort_keys=True)
    except ValueError:
        return None


def config_diff(
    earlier: Any, current: Any, path: str = ""
) -> List[Tuple[str, Any, Any]]:
    """Return (dotted path, earlier value, current value) for each changed leaf.

    The arguments are canonical structures. A list is one leaf.
    """
    if isinstance(earlier, dict) and isinstance(current, dict):
        changes = []
        keys = list(current) + [key for key in earlier if key not in current]
        for key in keys:
            changes += config_diff(
                earlier.get(key, _ABSENT),
                current.get(key, _ABSENT),
                f"{path}.{key}" if path else key,
            )
        return changes
    if earlier is _ABSENT or current is _ABSENT:
        return [(path, earlier, current)]
    if json.dumps(earlier, sort_keys=True) != json.dumps(current, sort_keys=True):
        return [(path, earlier, current)]
    return []


def format_config_value(value: Any) -> str:
    """Return the JSON text of a value, with each float as short float32 text."""
    if value is _ABSENT:
        return ABSENT_TEXT
    if isinstance(value, float):
        return str(np.float32(value))
    if isinstance(value, list):
        return "[" + ", ".join(format_config_value(item) for item in value) + "]"
    if isinstance(value, dict):
        items = [
            f"{json.dumps(key)}: {format_config_value(item)}"
            for key, item in value.items()
        ]
        return "{" + ", ".join(items) + "}"
    return json.dumps(value)


def config_diff_lines(
    earlier_text: str, current_text: str, language: str = "json"
) -> Optional[List[str]]:
    """Return one "path: earlier -> current" line for each changed leaf.

    Return None when one of the two texts does not parse.
    """
    try:
        earlier = canonical_config(earlier_text, language)
        current = canonical_config(current_text, language)
    except ValueError:
        return None
    return [
        f"{path}: {format_config_value(old)} -> {format_config_value(new)}"
        for path, old, new in config_diff(earlier, current)
    ]
