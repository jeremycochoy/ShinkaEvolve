import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)

# The canonicalizer of a JSON task for the tests. As the canonical form of an
# rnd spec, it gives one key to a key at its default and to no key
CANONICALIZER_MODULE = '''
DEFAULT_LEVERAGE = 1


def spec_key(config):
    """The config, with the default leverage when it has no leverage."""
    config["model_configuration"].setdefault("leverage", DEFAULT_LEVERAGE)
    return config


def strict_spec_key(config):
    """spec_key. As a strict loader, it refuses a T that is not an int."""
    if not isinstance(config["model_configuration"]["T"], int):
        raise ValueError("model_configuration.T: the format needs an integer")
    return spec_key(config)
'''


@pytest.fixture
def canonicalizer_dir(tmp_path, monkeypatch):
    """A temporary directory with the module spec_keys.

    sys.path and sys.modules come back to their state before the test.
    """
    directory = tmp_path / "canonicalizer"
    directory.mkdir()
    (directory / "spec_keys.py").write_text(CANONICALIZER_MODULE, encoding="utf-8")
    monkeypatch.setattr(sys, "path", list(sys.path))
    yield str(directory)
    sys.modules.pop("spec_keys", None)
