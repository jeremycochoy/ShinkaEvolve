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


def _make_argv_capturing_claude(tmp_path: Path, payload: dict) -> tuple[Path, Path]:
    """Fake `claude` that dumps its argv to a file and prints ``payload`` as JSON."""
    argv_capture = tmp_path / "argv.json"
    wrapper = tmp_path / "wrap_claude.py"
    wrapper.write_text(
        "\n".join(
            [
                "import json, sys",
                f"argv_path = {str(argv_capture)!r}",
                "if '--version' in sys.argv:",
                "    print('fake claude 1.0'); raise SystemExit(0)",
                "with open(argv_path, 'w') as f:",
                "    f.write(json.dumps(sys.argv[1:]))",
                f"print(json.dumps({payload!r}))",
            ]
        ),
        encoding="utf-8",
    )
    return wrapper, argv_capture


def test_query_claude_cli_forwards_model_effort_and_hardening_flags(tmp_path, monkeypatch):
    payload = {"is_error": False, "result": "x", "total_cost_usd": 0, "usage": {"output_tokens": 1}}
    wrapper, argv_capture = _make_argv_capturing_claude(tmp_path, payload)
    monkeypatch.setenv(
        "SHINKA_CLAUDE_CLI_COMMAND",
        f"{shlex.quote(sys.executable)} {shlex.quote(str(wrapper))}",
    )
    monkeypatch.setenv("SHINKA_CLAUDE_CLI_TIMEOUT", "10")

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
    assert argv[argv.index("--output-format") + 1] == "json"
    assert argv[argv.index("--model") + 1] == "claude-fable-5"
    assert argv[argv.index("--effort") + 1] == "max"
    assert argv[argv.index("--system-prompt") + 1] == "you are helpful"
    # Isolation contract: every tool disabled, no ambient settings, permission
    # checks left ON (nothing to bypass since --tools "" leaves no tools).
    assert argv[argv.index("--tools") + 1] == ""
    assert argv[argv.index("--setting-sources") + 1] == ""
    assert "--dangerously-skip-permissions" not in argv


def test_query_claude_cli_passes_empty_system_prompt_when_caller_gives_none(
    tmp_path, monkeypatch
):
    """An empty ``system_msg`` must still land as ``--system-prompt ""``
    so the Claude Code default agent persona is fully replaced (a truthy
    check would silently fall through to the CLI's built-in system prompt).
    """
    payload = {"is_error": False, "result": "x", "total_cost_usd": 0, "usage": {"output_tokens": 1}}
    wrapper, argv_capture = _make_argv_capturing_claude(tmp_path, payload)
    monkeypatch.setenv(
        "SHINKA_CLAUDE_CLI_COMMAND",
        f"{shlex.quote(sys.executable)} {shlex.quote(str(wrapper))}",
    )
    monkeypatch.setenv("SHINKA_CLAUDE_CLI_TIMEOUT", "10")

    query_claude_cli(
        None,
        "claude-cli/claude-opus-4-8",
        "hello",
        "",
        [],
        output_model=None,
    )

    argv = json.loads(argv_capture.read_text())
    assert "--system-prompt" in argv
    assert argv[argv.index("--system-prompt") + 1] == ""


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


def _install_concurrency_counting_fake(monkeypatch):
    """Return a ``max_active`` container updated by each fake subprocess."""
    state = {"active": 0, "max_active": 0}

    class FakeProcess:
        returncode = 0

        async def communicate(self, _input=None):
            await asyncio.sleep(0.02)
            state["active"] -= 1
            payload = json.dumps(
                {
                    "is_error": False,
                    "result": "ok",
                    "total_cost_usd": 0.0,
                    "usage": {"output_tokens": 1},
                }
            ).encode()
            return (payload, b"")

        def kill(self):
            raise AssertionError("fake process should not time out")

    async def fake_create_subprocess_exec(*_args, **_kwargs):
        state["active"] += 1
        state["max_active"] = max(state["max_active"], state["active"])
        return FakeProcess()

    monkeypatch.setenv("SHINKA_CLAUDE_CLI_COMMAND", "claude")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    return state


async def _fire_n_async_queries(n: int):
    await asyncio.gather(
        *[
            query_claude_cli_async(
                None,
                "claude-cli/claude-opus-4-8",
                "u",
                "s",
                [],
                output_model=None,
            )
            for _ in range(n)
        ]
    )


def test_query_claude_cli_runs_async_calls_in_parallel_by_default(monkeypatch):
    """No env cap set -> each `claude -p` is an independent session and
    concurrent invocations proceed in parallel (no process-wide lock)."""
    monkeypatch.delenv("SHINKA_CLAUDE_CLI_MAX_CONCURRENCY", raising=False)
    state = _install_concurrency_counting_fake(monkeypatch)

    asyncio.run(_fire_n_async_queries(4))

    assert state["max_active"] > 1, (
        f"expected parallel execution but only {state['max_active']} subprocess "
        "ran at once"
    )


def test_query_claude_cli_caps_async_concurrency_when_env_set(monkeypatch):
    monkeypatch.setenv("SHINKA_CLAUDE_CLI_MAX_CONCURRENCY", "1")
    state = _install_concurrency_counting_fake(monkeypatch)

    asyncio.run(_fire_n_async_queries(4))

    assert state["max_active"] == 1


def test_query_claude_cli_semaphore_release_survives_cancellation(monkeypatch):
    """Regression: an ``async with`` acquire releases on cancellation, so a
    later capped query must still be able to run."""
    monkeypatch.setenv("SHINKA_CLAUDE_CLI_MAX_CONCURRENCY", "1")

    class NeverProcess:
        returncode = 0

        async def communicate(self, _input=None):
            await asyncio.sleep(60)
            raise AssertionError("should have been cancelled")

        def kill(self):
            pass

    class QuickProcess:
        returncode = 0

        async def communicate(self, _input=None):
            payload = json.dumps(
                {
                    "is_error": False,
                    "result": "ok",
                    "total_cost_usd": 0.0,
                    "usage": {"output_tokens": 1},
                }
            ).encode()
            return (payload, b"")

        def kill(self):
            pass

    calls = {"n": 0}

    async def fake_create_subprocess_exec(*_args, **_kwargs):
        calls["n"] += 1
        return NeverProcess() if calls["n"] == 1 else QuickProcess()

    monkeypatch.setenv("SHINKA_CLAUDE_CLI_COMMAND", "claude")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    async def scenario():
        first = asyncio.create_task(
            query_claude_cli_async(
                None, "claude-cli/claude-opus-4-8", "u", "s", [], output_model=None
            )
        )
        # Let `first` acquire the semaphore and start awaiting communicate().
        await asyncio.sleep(0.05)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        # The semaphore must have been released; the second call must proceed.
        result = await asyncio.wait_for(
            query_claude_cli_async(
                None, "claude-cli/claude-opus-4-8", "u", "s", [], output_model=None
            ),
            timeout=2.0,
        )
        assert result.content == "ok"

    asyncio.run(scenario())


def test_query_claude_cli_retries_transient_then_succeeds(tmp_path, monkeypatch):
    """First invocation fails with an ``Overloaded`` message, second succeeds.
    The wrapper's transient-retry loop must swallow the first and return the
    second result rather than raising."""
    state_file = tmp_path / "attempts.txt"
    state_file.write_text("0", encoding="utf-8")
    payload = {"is_error": False, "result": "ok", "total_cost_usd": 0, "usage": {"output_tokens": 1}}
    script = tmp_path / "flaky_claude.py"
    script.write_text(
        "\n".join(
            [
                "import json, sys",
                f"state = {str(state_file)!r}",
                "if '--version' in sys.argv:",
                "    print('fake'); raise SystemExit(0)",
                "with open(state) as f: n = int(f.read())",
                "with open(state, 'w') as f: f.write(str(n + 1))",
                "if n == 0:",
                "    sys.stderr.write('Overloaded\\n'); raise SystemExit(1)",
                f"print(json.dumps({payload!r}))",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "SHINKA_CLAUDE_CLI_COMMAND",
        f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}",
    )
    monkeypatch.setenv("SHINKA_CLAUDE_CLI_TIMEOUT", "10")
    # Speed the retry backoff so the test doesn't sleep several seconds.
    import shinka.llm.providers.claude_cli as claude_cli_mod
    monkeypatch.setattr(claude_cli_mod, "_TRANSIENT_BACKOFF_SECONDS", 0.0)

    result = query_claude_cli(
        None,
        "claude-cli/claude-opus-4-8",
        "u",
        "s",
        [],
        output_model=None,
    )

    assert result.content == "ok"
    assert state_file.read_text() == "2"


def test_query_claude_cli_rejects_empty_stdout(tmp_path, monkeypatch):
    script = tmp_path / "silent_claude.py"
    script.write_text(
        "\n".join(
            [
                "import sys",
                "if '--version' in sys.argv:",
                "    print('fake'); raise SystemExit(0)",
                "raise SystemExit(0)",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "SHINKA_CLAUDE_CLI_COMMAND",
        f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}",
    )
    monkeypatch.setenv("SHINKA_CLAUDE_CLI_TIMEOUT", "10")

    with pytest.raises(ValueError, match="stdout was empty"):
        query_claude_cli(
            None,
            "claude-cli/claude-opus-4-8",
            "u",
            "s",
            [],
            output_model=None,
        )


def test_query_claude_cli_raises_on_non_zero_exit(tmp_path, monkeypatch):
    script = tmp_path / "failing_claude.py"
    script.write_text(
        "\n".join(
            [
                "import sys",
                "if '--version' in sys.argv:",
                "    print('fake'); raise SystemExit(0)",
                "sys.stderr.write('boom: internal error\\n')",
                "raise SystemExit(2)",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "SHINKA_CLAUDE_CLI_COMMAND",
        f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}",
    )
    monkeypatch.setenv("SHINKA_CLAUDE_CLI_TIMEOUT", "10")

    with pytest.raises(RuntimeError, match="boom: internal error"):
        query_claude_cli(
            None,
            "claude-cli/claude-opus-4-8",
            "u",
            "s",
            [],
            output_model=None,
        )


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
