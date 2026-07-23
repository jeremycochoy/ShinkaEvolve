from __future__ import annotations

import asyncio
import json
import shlex
import stat
import sys
from pathlib import Path

import pytest

from shinka.llm.client import get_async_client_llm, get_client_llm
from shinka.llm.kwargs import sample_model_kwargs
from shinka.llm.providers.claude_cli import (
    ClaudeCliModel,
    parse_claude_cli_model,
    query_claude_cli,
    query_claude_cli_async,
)
from shinka.llm.providers.model_resolver import resolve_model_backend
from shinka.model_availability import validate_model_env_access


def _make_fake_claude(tmp_path: Path, *, payload: dict) -> Path:
    """Write a python script that mimics `claude -p --output-format json`.

    The script:
    - responds to `--version` with a static string
    - otherwise reads stdin, echoes the constructed argv to stderr for
      inspection by tests, then prints the given JSON payload on stdout.
    """
    script = tmp_path / "fake_claude.py"
    script.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "import json",
                "import sys",
                "if '--version' in sys.argv:",
                "    print('claude 1.0.0 (fake)')",
                "    raise SystemExit(0)",
                "prompt = sys.stdin.read()",
                "print('ARGV=' + json.dumps(sys.argv[1:]), file=sys.stderr)",
                "print('PROMPT=' + prompt, file=sys.stderr)",
                f"print(json.dumps({payload!r}))",
            ]
        ),
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


def _fake_claude_command(script: Path) -> str:
    return f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}"


def test_parse_model_bare():
    parsed = parse_claude_cli_model("claude-cli/claude-opus-4-8")
    assert parsed == ClaudeCliModel(model="claude-opus-4-8", effort=None)


def test_parse_model_with_effort():
    parsed = parse_claude_cli_model("claude-cli/claude-opus-4-8?effort=low")
    assert parsed == ClaudeCliModel(model="claude-opus-4-8", effort="low")


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_parse_all_valid_efforts(effort: str):
    parsed = parse_claude_cli_model(f"claude-cli/claude-fable-5?effort={effort}")
    assert parsed.effort == effort


def test_parse_rejects_empty_model():
    with pytest.raises(ValueError, match="Claude CLI model name is required"):
        parse_claude_cli_model("claude-cli/")


def test_parse_rejects_unknown_effort():
    with pytest.raises(ValueError, match="Unsupported claude-cli effort"):
        parse_claude_cli_model("claude-cli/claude-opus-4-8?effort=bogus")


def test_parse_rejects_unknown_query_param():
    with pytest.raises(ValueError, match="Unsupported claude-cli query parameter"):
        parse_claude_cli_model("claude-cli/claude-opus-4-8?bogus=1")


def test_parse_rejects_duplicate_effort():
    with pytest.raises(ValueError, match="effort only once"):
        parse_claude_cli_model("claude-cli/claude-opus-4-8?effort=low&effort=high")


def test_parse_rejects_wrong_prefix():
    with pytest.raises(ValueError, match="must start with 'claude-cli/'"):
        parse_claude_cli_model("openrouter/foo")


def test_resolve_model_backend_routes_to_claude_cli():
    resolved = resolve_model_backend("claude-cli/claude-opus-4-8?effort=low")
    assert resolved.provider == "claude_cli"
    assert resolved.api_model_name == "claude-cli/claude-opus-4-8?effort=low"
    assert resolved.base_url is None


def test_resolve_model_backend_rejects_bare_prefix():
    with pytest.raises(ValueError, match="Claude CLI model name is missing"):
        resolve_model_backend("claude-cli/")


def test_get_client_llm_returns_none_for_claude_cli():
    client, name, provider = get_client_llm("claude-cli/claude-opus-4-8?effort=low")
    assert client is None
    assert provider == "claude_cli"
    assert name == "claude-cli/claude-opus-4-8?effort=low"


def test_get_async_client_llm_returns_none_for_claude_cli():
    client, name, provider = get_async_client_llm("claude-cli/claude-fable-5")
    assert client is None
    assert provider == "claude_cli"
    assert name == "claude-cli/claude-fable-5"


def test_claude_cli_kwargs_skip_api_only_parameters():
    kwargs = sample_model_kwargs(
        model_names=["claude-cli/claude-opus-4-8?effort=low"],
        temperatures=[0.0, 1.0],
        max_tokens=[128],
        reasoning_efforts=["high"],
    )
    assert kwargs == {"model_name": "claude-cli/claude-opus-4-8?effort=low"}


def test_query_claude_cli_invokes_command_and_parses_json(tmp_path, monkeypatch):
    payload = {
        "type": "result",
        "is_error": False,
        "result": "hello from claude",
        "total_cost_usd": 0.0123,
        "usage": {
            "input_tokens": 9,
            "cache_creation_input_tokens": 100,
            "cache_read_input_tokens": 200,
            "output_tokens": 42,
        },
    }
    fake = _make_fake_claude(tmp_path, payload=payload)
    monkeypatch.setenv("SHINKA_CLAUDE_CLI_COMMAND", _fake_claude_command(fake))
    monkeypatch.setenv("SHINKA_CLAUDE_CLI_TIMEOUT", "10")

    result = query_claude_cli(
        None,
        "claude-cli/claude-opus-4-8?effort=low",
        "user request",
        "system instructions",
        [],
        output_model=None,
    )

    assert result.content == "hello from claude"
    assert result.model_name == "claude-cli/claude-opus-4-8?effort=low"
    assert result.input_tokens == 9 + 100 + 200
    assert result.output_tokens == 42
    assert result.cost == pytest.approx(0.0123)


def test_query_claude_cli_forwards_model_and_effort_flags(tmp_path, monkeypatch):
    payload = {"is_error": False, "result": "x", "total_cost_usd": 0, "usage": {"output_tokens": 1}}
    fake = _make_fake_claude(tmp_path, payload=payload)
    monkeypatch.setenv("SHINKA_CLAUDE_CLI_COMMAND", _fake_claude_command(fake))
    monkeypatch.setenv("SHINKA_CLAUDE_CLI_TIMEOUT", "10")
    argv_capture = tmp_path / "argv.txt"

    # Wrap the fake to capture argv to a file (stderr is discarded by the runner).
    wrapper = tmp_path / "wrap_claude.py"
    wrapper.write_text(
        "\n".join(
            [
                "import json, subprocess, sys",
                f"argv_path = {str(argv_capture)!r}",
                "if '--version' in sys.argv:",
                "    print('fake'); raise SystemExit(0)",
                f"with open(argv_path, 'w') as f:",
                "    f.write(json.dumps(sys.argv[1:]))",
                f"print(json.dumps({payload!r}))",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "SHINKA_CLAUDE_CLI_COMMAND",
        f"{shlex.quote(sys.executable)} {shlex.quote(str(wrapper))}",
    )

    query_claude_cli(
        None,
        "claude-cli/claude-fable-5?effort=max",
        "hello",
        "you are helpful",
        [],
        output_model=None,
    )

    argv = json.loads(argv_capture.read_text())
    assert "-p" in argv
    assert "--output-format" in argv and argv[argv.index("--output-format") + 1] == "json"
    assert argv[argv.index("--model") + 1] == "claude-fable-5"
    assert argv[argv.index("--effort") + 1] == "max"
    assert argv[argv.index("--system-prompt") + 1] == "you are helpful"


def test_query_claude_cli_rejects_structured_output(tmp_path, monkeypatch):
    from pydantic import BaseModel

    class Out(BaseModel):
        x: int

    with pytest.raises(ValueError, match="does not support structured output"):
        query_claude_cli(
            None,
            "claude-cli/claude-opus-4-8",
            "x",
            "y",
            [],
            output_model=Out,
        )


def test_query_claude_cli_surfaces_error_payload(tmp_path, monkeypatch):
    payload = {
        "is_error": True,
        "result": "quota exceeded",
        "total_cost_usd": 0,
        "usage": {},
    }
    fake = _make_fake_claude(tmp_path, payload=payload)
    monkeypatch.setenv("SHINKA_CLAUDE_CLI_COMMAND", _fake_claude_command(fake))
    monkeypatch.setenv("SHINKA_CLAUDE_CLI_TIMEOUT", "10")

    with pytest.raises(RuntimeError, match="quota exceeded"):
        query_claude_cli(
            None,
            "claude-cli/claude-opus-4-8",
            "x",
            "y",
            [],
            output_model=None,
        )


def test_query_claude_cli_serializes_async_calls(tmp_path, monkeypatch):
    active = 0
    max_active = 0

    class FakeProcess:
        returncode = 0

        async def communicate(self, _input=None):
            nonlocal active
            await asyncio.sleep(0.02)
            active -= 1
            payload = json.dumps({
                "is_error": False,
                "result": "ok",
                "total_cost_usd": 0.0,
                "usage": {"output_tokens": 1},
            }).encode()
            return (payload, b"")

        def kill(self):
            raise AssertionError("fake process should not time out")

    async def fake_create_subprocess_exec(*args, **kwargs):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        return FakeProcess()

    monkeypatch.setenv("SHINKA_CLAUDE_CLI_COMMAND", "claude")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    async def run_two():
        await asyncio.gather(
            query_claude_cli_async(
                None,
                "claude-cli/claude-opus-4-8",
                "u",
                "s",
                [],
                output_model=None,
            ),
            query_claude_cli_async(
                None,
                "claude-cli/claude-opus-4-8",
                "u",
                "s",
                [],
                output_model=None,
            ),
        )

    asyncio.run(run_two())
    assert max_active == 1


def test_validate_model_env_access_runs_claude_cli_check(tmp_path, monkeypatch):
    payload = {"is_error": False, "result": "ok", "total_cost_usd": 0, "usage": {}}
    fake = _make_fake_claude(tmp_path, payload=payload)
    monkeypatch.setenv("SHINKA_CLAUDE_CLI_COMMAND", _fake_claude_command(fake))
    monkeypatch.setenv("SHINKA_CLAUDE_CLI_TIMEOUT", "10")

    validate_model_env_access(llm_models=["claude-cli/claude-opus-4-8?effort=low"])


def test_validate_model_env_access_flags_missing_claude_cli(tmp_path, monkeypatch):
    monkeypatch.setenv("SHINKA_CLAUDE_CLI_COMMAND", "nonexistent-command-does-not-exist")
    monkeypatch.setenv("SHINKA_CLAUDE_CLI_TIMEOUT", "10")
    with pytest.raises(ValueError, match="claude-cli model.* are unavailable"):
        validate_model_env_access(llm_models=["claude-cli/claude-opus-4-8"])
