"""Claude Code CLI (`claude -p`) as a Shinka LLM provider.

Model naming: ``claude-cli/<model>[?effort=<level>]``.

Examples:
    ``claude-cli/claude-opus-4-8``
    ``claude-cli/claude-opus-4-8?effort=low``
    ``claude-cli/claude-opus-4-8?effort=max``
    ``claude-cli/claude-opus-4-7?effort=low``
    ``claude-cli/claude-fable-5?effort=low``

Effort levels accepted mirror the CLI: ``low``, ``medium``, ``high``, ``xhigh``,
``max``. When ``effort`` is omitted, the CLI's default is used.

The wrapper shells out to ``claude -p`` with ``--output-format json`` and reads
the resulting JSON payload for content and token usage. Concurrent async calls
are serialised through a process-wide lock (Claude Code holds session-scoped
state so parallel invocations from the same account are not safe).

Authentication uses whatever ``claude`` itself resolves at runtime (typically
the persisted OAuth login for a Claude Code subscription). Any inherited
``ANTHROPIC_API_KEY`` / ``CLAUDE_CODE_OAUTH_TOKEN`` env vars are stripped from
the subprocess environment so the CLI cannot silently switch billing paths.
"""
from __future__ import annotations

import asyncio
import json
import os
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs

from pydantic import BaseModel

from shinka.llm.constants import TIMEOUT

from .result import QueryResult

CLAUDE_CLI_PREFIX = "claude-cli/"
DEFAULT_CLAUDE_CLI_COMMAND = "claude"
CLAUDE_CLI_COMMAND_ENV = "SHINKA_CLAUDE_CLI_COMMAND"
CLAUDE_CLI_TIMEOUT_ENV = "SHINKA_CLAUDE_CLI_TIMEOUT"

_VALID_EFFORTS = {"low", "medium", "high", "xhigh", "max"}
_THREAD_LOCK = threading.Lock()
_TRANSIENT_RETRIES = 2
_TRANSIENT_BACKOFF_SECONDS = 3.0


@dataclass(frozen=True)
class ClaudeCliModel:
    model: str
    effort: str | None = None


def claude_cli_command_prefix() -> list[str]:
    raw = os.getenv(CLAUDE_CLI_COMMAND_ENV, DEFAULT_CLAUDE_CLI_COMMAND).strip()
    if not raw:
        raise ValueError(f"{CLAUDE_CLI_COMMAND_ENV} cannot be empty.")
    return shlex.split(raw)


def claude_cli_timeout() -> float:
    raw = os.getenv(CLAUDE_CLI_TIMEOUT_ENV)
    if raw is None:
        return float(TIMEOUT)
    value = float(raw)
    if value <= 0:
        raise ValueError(f"{CLAUDE_CLI_TIMEOUT_ENV} must be > 0.")
    return value


def parse_claude_cli_model(model_name: str) -> ClaudeCliModel:
    if not model_name.startswith(CLAUDE_CLI_PREFIX):
        raise ValueError(
            f"Claude CLI model must start with '{CLAUDE_CLI_PREFIX}': {model_name}"
        )
    body = model_name.split(CLAUDE_CLI_PREFIX, 1)[1]
    route, _, query = body.partition("?")
    if not route:
        raise ValueError("Claude CLI model name is required after 'claude-cli/'.")

    parsed = parse_qs(query, keep_blank_values=True)
    unknown = sorted(set(parsed) - {"effort"})
    if unknown:
        raise ValueError(
            f"Unsupported claude-cli query parameter(s): {unknown}"
        )

    efforts = parsed.get("effort", [])
    if len(efforts) > 1:
        raise ValueError("Claude CLI model may specify effort only once.")
    effort = efforts[0] if efforts else None
    if effort is not None and effort not in _VALID_EFFORTS:
        raise ValueError(
            f"Unsupported claude-cli effort '{effort}'. "
            f"Expected one of {sorted(_VALID_EFFORTS)}."
        )

    return ClaudeCliModel(model=route, effort=effort)


def check_claude_cli_available() -> None:
    cmd = [*claude_cli_command_prefix(), "--version"]
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=min(claude_cli_timeout(), 30.0),
            check=False,
        )
    except FileNotFoundError as exc:
        raise ValueError(f"claude CLI not found: {cmd[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise ValueError("claude CLI --version check timed out.") from exc
    if completed.returncode != 0:
        detail = (
            completed.stderr.strip()
            or completed.stdout.strip()
            or f"exit code {completed.returncode}"
        )
        raise ValueError(f"claude CLI availability check failed: {detail}")


def _render_prompt(msg: str, msg_history: list[dict]) -> str:
    if not msg_history:
        return msg
    history_text = json.dumps(msg_history, indent=2, ensure_ascii=False)
    return (
        "# Previous Messages\n\n"
        f"{history_text}\n\n"
        "# User Request\n\n"
        f"{msg}\n"
    )


def _build_command(model: ClaudeCliModel, system_msg: str) -> list[str]:
    cmd = [
        *claude_cli_command_prefix(),
        "-p",
        "--output-format",
        "json",
        "--dangerously-skip-permissions",
    ]
    if system_msg:
        cmd.extend(["--system-prompt", system_msg])
    if model.model:
        cmd.extend(["--model", model.model])
    if model.effort:
        cmd.extend(["--effort", model.effort])
    return cmd


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)
    env.pop("CLAUDE_CODE_OAUTH_TOKEN", None)
    return env


def _is_transient_error(completed: subprocess.CompletedProcess) -> bool:
    if completed.returncode == 0:
        return False
    output = f"{completed.stderr}\n{completed.stdout}"
    return (
        "Credit balance is too low" in output
        or "Overloaded" in output
        or "rate_limit" in output.lower()
    )


def _run_sync(command: list[str], prompt: str) -> subprocess.CompletedProcess:
    attempts = _TRANSIENT_RETRIES + 1
    for attempt in range(attempts):
        completed = subprocess.run(
            command,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=claude_cli_timeout(),
            check=False,
            env=_subprocess_env(),
        )
        if not _is_transient_error(completed):
            return completed
        if attempt < attempts - 1:
            time.sleep(_TRANSIENT_BACKOFF_SECONDS * (attempt + 1))
    return completed


async def _run_async(command: list[str], prompt: str) -> subprocess.CompletedProcess:
    attempts = _TRANSIENT_RETRIES + 1
    for attempt in range(attempts):
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=_subprocess_env(),
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(prompt.encode("utf-8")),
                timeout=claude_cli_timeout(),
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.communicate()
            raise
        completed = subprocess.CompletedProcess(
            args=command,
            returncode=process.returncode,
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
        )
        if not _is_transient_error(completed):
            return completed
        if attempt < attempts - 1:
            await asyncio.sleep(_TRANSIENT_BACKOFF_SECONDS * (attempt + 1))
    return completed


def _parse_response(stdout: str) -> tuple[str, dict[str, Any]]:
    trimmed = stdout.strip()
    if not trimmed:
        raise ValueError("claude CLI stdout was empty.")
    last_line = trimmed.splitlines()[-1]
    try:
        payload = json.loads(last_line)
    except json.JSONDecodeError as exc:
        raise ValueError("claude CLI stdout was not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError("claude CLI JSON output must be an object.")
    if payload.get("is_error"):
        raise RuntimeError(
            f"claude CLI reported error: {payload.get('result') or payload}"
        )
    result = payload.get("result", "")
    if not isinstance(result, str) or not result:
        raise ValueError("claude CLI response missing non-empty 'result' field.")
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    cost = payload.get("total_cost_usd", 0.0)
    try:
        cost_value = float(cost) if cost is not None else 0.0
    except (TypeError, ValueError):
        cost_value = 0.0
    return result, {"usage": usage or {}, "cost": cost_value}


def _int_field(source: dict[str, Any], *names: str) -> int:
    for name in names:
        value = source.get(name)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0


def _query_result(
    *,
    content: str,
    meta: dict[str, Any],
    model: str,
    msg: str,
    system_msg: str,
    msg_history: list[dict],
    kwargs: dict[str, Any],
    model_posteriors: dict[str, float] | None,
) -> QueryResult:
    usage = meta["usage"]
    new_msg_history = [
        *msg_history,
        {"role": "user", "content": msg},
        {"role": "assistant", "content": content},
    ]
    input_tokens = (
        _int_field(usage, "input_tokens", "inputTokens")
        + _int_field(usage, "cache_creation_input_tokens", "cacheCreationInputTokens")
        + _int_field(usage, "cache_read_input_tokens", "cacheReadInputTokens")
    )
    output_tokens = _int_field(usage, "output_tokens", "outputTokens")
    thinking_tokens = _int_field(
        usage, "reasoning_tokens", "thinking_tokens", "reasoningOutputTokens"
    )
    return QueryResult(
        content=content,
        msg=msg,
        system_msg=system_msg,
        new_msg_history=new_msg_history,
        model_name=model,
        kwargs=kwargs,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        thinking_tokens=thinking_tokens,
        cost=meta["cost"],
        input_cost=0.0,
        output_cost=0.0,
        model_posteriors=model_posteriors,
        num_total_queries=1,
    )


def query_claude_cli(
    client,
    model,
    msg,
    system_msg,
    msg_history,
    output_model: BaseModel | None,
    model_posteriors=None,
    **kwargs,
) -> QueryResult:
    if output_model is not None:
        raise ValueError("claude-cli does not support structured output.")

    parsed = parse_claude_cli_model(model)
    prompt = _render_prompt(msg, msg_history)
    command = _build_command(parsed, system_msg)

    try:
        with _THREAD_LOCK:
            completed = _run_sync(command, prompt)
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"claude CLI query timed out after {claude_cli_timeout()}s."
        ) from exc

    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"claude CLI query failed: {detail}")

    content, meta = _parse_response(completed.stdout)
    return _query_result(
        content=content,
        meta=meta,
        model=model,
        msg=msg,
        system_msg=system_msg,
        msg_history=msg_history,
        kwargs={**kwargs, "model_name": model},
        model_posteriors=model_posteriors,
    )


async def _acquire_cli_lock_async() -> None:
    await asyncio.to_thread(_THREAD_LOCK.acquire)


async def query_claude_cli_async(
    client,
    model,
    msg,
    system_msg,
    msg_history,
    output_model: BaseModel | None,
    model_posteriors=None,
    **kwargs,
) -> QueryResult:
    if output_model is not None:
        raise ValueError("claude-cli does not support structured output.")

    parsed = parse_claude_cli_model(model)
    prompt = _render_prompt(msg, msg_history)
    command = _build_command(parsed, system_msg)

    await _acquire_cli_lock_async()
    try:
        try:
            completed = await _run_async(command, prompt)
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"claude CLI query timed out after {claude_cli_timeout()}s."
            ) from exc
    finally:
        _THREAD_LOCK.release()

    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"claude CLI query failed: {detail}")

    content, meta = _parse_response(completed.stdout)
    return _query_result(
        content=content,
        meta=meta,
        model=model,
        msg=msg,
        system_msg=system_msg,
        msg_history=msg_history,
        kwargs={**kwargs, "model_name": model},
        model_posteriors=model_posteriors,
    )
