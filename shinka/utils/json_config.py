"""Canonical structure, key and diff of the config of a JSON task.

The async runner rejects a proposal when its key is the key of an earlier
config, and the proposal prompt gives an earlier config as its changed keys.
A config parses as JSON5, as the task evaluator does: a comment or a trailing
comma does not change the config.
Each float becomes its float32 value: 0.0010000000474974513 and 0.001 are the
same value. An int stays exact, thus an int and a float are different values.
A bool is never equal to a number.
A task can give a canonicalizer: a function of the parsed config that gives
the key. Then two configs of the same run can have the same key, for example
a config that writes a key at its default and a config without the key.
"""

import importlib
import json
import os
import sys
from typing import Any, Callable, List, Optional, Tuple

import json5
import numpy as np

ABSENT_TEXT = "(absent)"
# The value of a key that one side of a diff does not have
_ABSENT = object()

# A function of the parsed config. The JSON text of its value is the key
Canonicalizer = Callable[[Any], Any]


class CanonicalizerError(Exception):
    """The canonicalizer of a JSON task refuses a config."""


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


def canonical_config(text: str) -> Any:
    """Parse a config as JSON5, with float32 floats.

    Raise ValueError when the text does not parse.
    """
    return to_float32(json5.loads(text))


def load_canonicalizer(target: str, path: Optional[str] = None) -> Canonicalizer:
    """Import the function of a "module:function" target.

    The path goes first in sys.path before the import. A relative path starts
    at the working directory. Raise ValueError, with the target in the
    message, when the import fails or gives no function.
    """
    module_name, _, function_name = target.partition(":")
    if not module_name or not function_name:
        raise ValueError(f"json_canonicalizer {target!r}: write module:function")
    if path is not None and os.path.abspath(path) not in sys.path:
        sys.path.insert(0, os.path.abspath(path))
    try:
        function = getattr(importlib.import_module(module_name), function_name)
    except Exception as error:
        raise ValueError(
            f"json_canonicalizer {target!r}: {type(error).__name__}: {error}"
        ) from error
    if not callable(function):
        raise ValueError(f"json_canonicalizer {target!r}: not a function")
    return function


def config_key(
    text: str, canonicalizer: Optional[Canonicalizer] = None
) -> Optional[str]:
    """Return the key of a config, or None when the text does not parse.

    Without a canonicalizer, the key is the JSON text of the canonical
    structure. With a canonicalizer, the key is the JSON text of its value for
    the config that json5.loads gives. Raise CanonicalizerError when the
    canonicalizer raises, or when its value has no JSON text.
    """
    try:
        config = json5.loads(text)
    except ValueError:
        return None
    if canonicalizer is None:
        return json.dumps(to_float32(config), sort_keys=True)
    try:
        return json.dumps(canonicalizer(config), sort_keys=True)
    except Exception as error:
        raise CanonicalizerError(f"{type(error).__name__}: {error}") from error


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


def config_diff_lines(earlier_text: str, current_text: str) -> Optional[List[str]]:
    """Return one "path: earlier -> current" line for each changed leaf.

    Return None when one of the two texts does not parse.
    """
    try:
        earlier = canonical_config(earlier_text)
        current = canonical_config(current_text)
    except ValueError:
        return None
    return [
        f"{path}: {format_config_value(old)} -> {format_config_value(new)}"
        for path, old, new in config_diff(earlier, current)
    ]
