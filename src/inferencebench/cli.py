"""Inspect SWE adapters with shared continuation, tools, and model context settings."""

import hashlib
import inspect
import json
import os
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, cast

import inspect_swe
from inspect_ai.agent import Agent, AgentState, BridgedToolsSpec, agent
from inspect_ai.model import ChatMessageUser, GenerateInput, get_model, get_model_info
from inspect_ai.model._generate_config import active_generate_config
from inspect_ai.util import (
    ExecRemoteAwaitableOptions,
    ExecRemoteProcess,
    ExecRemoteStreamingOptions,
    ExecResult,
    concurrency,
    sandbox,
    store,
)
from inspect_ai.util._sandbox._cli import SANDBOX_CLI

from inferencebench.reminders import cli_reminders, nudge, with_deadline
from inferencebench.run_config import load_config
from inferencebench.tools import web_search

DEFAULT_AGENT_ARGS = load_config("run_configs/claude_code.yaml")["solver"]["args"]
CLI_HARNESSES = ("claude_code", "codex_cli", "gemini_cli", "kimi_code", "opencode")

# OpenCode's native compaction policy verified in ExploitBench: advertise half the context and prune early.
OPENCODE_CONTEXT_FRACTION = 2
OPENCODE_MIN_RESERVED_TOKENS = 8_000
OPENCODE_MIN_PRESERVE_RECENT_TOKENS = 4_000
OPENCODE_TAIL_TURNS = 8
OPENCODE_TOOL_OUTPUT_MAX_LINES = 1_000
OPENCODE_TOOL_OUTPUT_MAX_BYTES = 20 * 1024

# Official release archives with published digests, staged so concurrent samples skip GitHub's metadata API.
CLI_ARCHIVES = {
    ("codex_cli", "0.154.0"): (
        "https://github.com/openai/codex/releases/download/rust-v0.154.0/codex-package-x86_64-unknown-linux-musl.tar.gz",
        "fc6e3e3b85f2cf7d664520ee5c66a7fe4aa12bae7d46834f47e2f165fd0d6f78",
    ),
}


async def file_cli_prompts(state):
    """Keep pending user instructions out of argv so server-oriented pkill cannot match them."""
    for message in reversed(state.messages):
        if message.role == "assistant":
            break
        if message.role == "user" and not message.text.startswith("Read /tmp/inferencebench-input-"):
            path = f"/tmp/inferencebench-input-{message.id}.txt"
            await sandbox().write_file(path, message.text)
            message.content = f"Read {path} and follow its instructions."


@agent
def cli_agent(
    harness: str,
    harness_args: dict[str, Any] | None = None,
    nudge_prompt: bool = DEFAULT_AGENT_ARGS["nudge_prompt"],
    token_budget_reminder: bool = DEFAULT_AGENT_ARGS["token_budget_reminder"],
    web_search_args: dict = DEFAULT_AGENT_ARGS["web_search_args"],
    cli_poll_timeout: int | None = DEFAULT_AGENT_ARGS["cli_poll_timeout"],
) -> Agent:
    """Resume the same native CLI session after early exits, using Inspect for all model calls."""
    if harness not in CLI_HARNESSES:
        raise ValueError(
            f"Unknown CLI harness {harness!r}; choose from {CLI_HARNESSES}"
        )
    if cli_poll_timeout is not None and (type(cli_poll_timeout) is not int or cli_poll_timeout <= 0):
        raise ValueError("cli_poll_timeout must be a positive integer or null")
    factory = getattr(inspect_swe, harness)
    args = dict(harness_args or {})
    reserved = {"user", "cwd", "sandbox", "bridged_tools", "filter"}.intersection(args)
    if reserved:
        raise ValueError(
            f"InferenceBench supplies these harness arguments: {sorted(reserved)}"
        )
    unknown = args.keys() - inspect.signature(factory).parameters.keys()
    if unknown:
        raise TypeError(f"Unknown {harness} arguments: {sorted(unknown)}")
    # Native multi-attempt agents invoke the destructive final scorer between attempts.
    if args.get("attempts", 1) != 1:
        raise ValueError(
            "CLI attempts must be 1; continuation resumes the existing session"
        )
    if harness == "claude_code":
        args["disallowed_tools"] = list(
            dict.fromkeys([*(args.get("disallowed_tools") or []), "WebSearch"])
        )

    async def model_filter(model, messages, tools, tool_choice, config):
        """Add current budget information to task requests through the native API bridge."""
        return cli_reminders(
            GenerateInput(
                input=messages, tools=tools, tool_choice=tool_choice, config=config
            ),
            token_budget_reminder,
        )

    async def execute(state: AgentState) -> AgentState:
        """Create one CLI and retain its session across continuation messages."""
        await stage_cli_archive(harness, args.get("version"))
        cli = factory(
            cwd="/home/agent/task",
            user="root",
            sandbox="default",
            bridged_tools=[
                BridgedToolsSpec(
                    name="inferencebench", tools=[web_search(**web_search_args)]
                )
            ],
            filter=model_filter,
            **context_args(harness, args),
        )
        while True:
            if harness == "claude_code":
                await file_cli_prompts(state)
            with remote_poll_timeout(cli_poll_timeout, harness):
                state = await cli(state)
            continuation = nudge(nudge_prompt)
            if continuation is False:
                return state
            state.messages.append(ChatMessageUser(content=str(continuation)))

    return with_deadline(execute)


@contextmanager
def remote_poll_timeout(timeout: int | None, harness: str) -> Iterator[None]:
    """Give this sample's sandbox longer remote-process polls; native adapters omit the RPC timeout and the model proxy uses 600 seconds."""
    if timeout is None:
        yield
        return
    environment = sandbox("default")
    original = environment.exec_remote
    instance_override = "exec_remote" in vars(environment)
    call = cast(Callable[..., Awaitable[ExecRemoteProcess | ExecResult[str]]], original)

    async def exec_remote(cmd, options=None, *, stream=True):
        """Set the poll timeout on CLI commands and, for OpenCode, on the bridge's model proxy."""
        if options is None:
            options = ExecRemoteStreamingOptions() if stream else ExecRemoteAwaitableOptions()
        if options.poll_timeout is None:
            options = replace(options, poll_timeout=timeout)
        if harness == "opencode" and cmd == [SANDBOX_CLI, "model_proxy"] and isinstance(options, ExecRemoteStreamingOptions):
            options = replace(options, poll_timeout=timeout)
        return await call(cmd, options, stream=stream)

    # Patch the sample-local instance, never the shared framework class.
    setattr(environment, "exec_remote", exec_remote)
    try:
        yield
    finally:
        if instance_override:
            setattr(environment, "exec_remote", original)
        else:
            delattr(environment, "exec_remote")


async def stage_cli_archive(harness: str, version: str | None) -> None:
    """Place the pinned official release archive in Inspect SWE's cache after checking its digest."""
    if (harness, version) not in CLI_ARCHIVES:
        return
    from inspect_swe._codex_cli.agentbinary import codex_cli_binary_source
    from inspect_swe._util.download import download_file
    from inspect_swe._util.sandbox import detect_sandbox_platform

    platform = await detect_sandbox_platform(sandbox())
    if platform != "linux-x64":
        return
    path = codex_cli_binary_source().cached_package_path(version, platform)
    url, checksum = CLI_ARCHIVES[(harness, version)]
    async with concurrency(f"inferencebench-{harness}-archive", 1, visible=False):
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == checksum:
            return
        data = await download_file(url)
        if hashlib.sha256(data).hexdigest() != checksum:
            raise ValueError(f"CLI release checksum mismatch: {url}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(data)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def context_args(harness: str, args: dict[str, Any]) -> dict[str, Any]:
    """Pass the served model's context and output limits to each CLI's native configuration."""
    args = dict(args)
    model = get_model()
    info = get_model_info(model)
    context = info.context_length if info else None
    output = model.config.merge(active_generate_config()).max_tokens
    # Preparation records the served model for the sandbox; explicit harness env wins.
    env = {**(store().get("workspace_env") or {}), **(args.get("env") or {})}
    if harness == "claude_code":
        if context:
            env.setdefault("CLAUDE_CODE_MAX_CONTEXT_TOKENS", str(context))
        if output is not None:
            env.setdefault("CLAUDE_CODE_MAX_OUTPUT_TOKENS", str(output))
        env.setdefault("CLAUDE_CODE_TOTAL_TOKENS_REMINDER", "off")
    elif harness == "codex_cli" and context:
        args["config_overrides"] = {
            "model_context_window": str(context),
            **(args.get("config_overrides") or {}),
        }
    elif harness == "kimi_code" and context:
        args.setdefault("max_context_size", context)
    elif harness == "opencode" and context:
        identity = args.setdefault("opencode_model", args.get("model") or str(model))
        provider, model_id = identity.split("/", 1)
        config = json.loads(env.get("OPENCODE_CONFIG_CONTENT", "{}"))
        config.setdefault("small_model", identity)
        advertised = max(1, context // OPENCODE_CONTEXT_FRACTION)
        compaction = config.setdefault("compaction", {})
        compaction.setdefault("auto", True)
        compaction.setdefault("prune", True)
        compaction.setdefault("tail_turns", OPENCODE_TAIL_TURNS)
        compaction.setdefault("preserve_recent_tokens", min(max(OPENCODE_MIN_PRESERVE_RECENT_TOKENS, context // 8), max(1, advertised // 4)))
        compaction.setdefault("reserved", min(max(OPENCODE_MIN_RESERVED_TOKENS, context // 4), max(1, advertised // 2)))
        tool_output = config.setdefault("tool_output", {})
        tool_output.setdefault("max_lines", OPENCODE_TOOL_OUTPUT_MAX_LINES)
        tool_output.setdefault("max_bytes", OPENCODE_TOOL_OUTPUT_MAX_BYTES)
        provider_config = config.setdefault("provider", {}).setdefault(provider, {})
        options = provider_config.setdefault("options", {})
        options.setdefault("apiKey", "sk-none")
        if provider == "openrouter":
            options.setdefault("compatibility", "strict")
        # OpenCode's provider timers are milliseconds; keep them at Inspect's attempt timeout.
        attempt_timeout = model.config.merge(active_generate_config()).attempt_timeout
        if attempt_timeout is not None:
            for key in ["timeout", "headerTimeout", "chunkTimeout"]:
                options.setdefault(key, int(attempt_timeout * 1000))
        limits = (
            provider_config.setdefault("models", {})
            .setdefault(model_id, {})
            .setdefault("limit", {})
        )
        limits.setdefault("context", advertised)
        output = output if output is not None else info.output_tokens
        if output is None:
            raise ValueError(
                "OpenCode requires max_tokens or model metadata output_tokens"
            )
        limits.setdefault("output", output)
        env["OPENCODE_CONFIG_CONTENT"] = json.dumps(config)
    if env:
        args["env"] = env
    return args
