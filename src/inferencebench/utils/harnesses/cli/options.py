import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

from inspect_ai.model import get_model, get_model_info
from inspect_ai.model._generate_config import active_generate_config
from inspect_ai.util import sandbox

CLI_HARNESSES = ("claude_code", "codex_cli", "gemini_cli", "kimi_code", "opencode")
OPENCODE_CONTEXT_FRACTION = 2
OPENCODE_MIN_RESERVED_TOKENS = 8_000
OPENCODE_MIN_PRESERVE_RECENT_TOKENS = 4_000
OPENCODE_TAIL_TURNS = 8
OPENCODE_TOOL_OUTPUT_MAX_LINES = 1_000
OPENCODE_TOOL_OUTPUT_MAX_BYTES = 20 * 1024


async def _gemini_timeout_args(
    args: dict[str, Any], attempt_timeout: int | None
) -> dict[str, Any]:
    """Apply the API attempt timeout to Gemini's native response-header timeout."""
    env = dict(args.get("env") or {})
    if attempt_timeout is None or "GEMINI_EXP" in env:
        return args
    # Gemini CLI's DEFAULT_REQUEST_TIMEOUT experiment flag takes seconds.
    # Its native loader supports GEMINI_EXP even with API-key authentication.
    path = f"/tmp/inspect-gemini-{uuid4().hex}.json"
    await sandbox("default").write_file(
        path,
        json.dumps(
            {"flags": [{"flagId": "45773134", "intValue": str(attempt_timeout)}]}
        ),
    )
    return {**args, "env": {**env, "GEMINI_EXP": path}}


def _opencode_timeout_args(
    args: dict[str, Any], attempt_timeout: float | None
) -> dict[str, Any]:
    """Keep OpenCode's provider timers aligned with Inspect's attempt timeout."""
    if attempt_timeout is None:
        return args
    env = dict(args.get("env") or {})
    raw_config = env.get("OPENCODE_CONFIG_CONTENT")
    config = json.loads(raw_config) if raw_config else {}
    if not isinstance(config, dict):
        raise ValueError("OPENCODE_CONFIG_CONTENT must contain a JSON object")
    model = str(args.get("opencode_model", "anthropic/claude-sonnet-4-5"))
    provider_id = model.split("/", 1)[0]
    providers = config.setdefault("provider", {})
    if not isinstance(providers, dict):
        raise ValueError("OpenCode provider config must be a JSON object")
    provider = providers.setdefault(provider_id, {})
    if not isinstance(provider, dict):
        raise ValueError(f"OpenCode provider {provider_id!r} must be a JSON object")
    provider_options = provider.setdefault("options", {})
    if not isinstance(provider_options, dict):
        raise ValueError(
            f"OpenCode provider {provider_id!r} options must be a JSON object"
        )
    timeout_ms = int(attempt_timeout * 1000)
    provider_options.setdefault("timeout", timeout_ms)
    provider_options.setdefault("headerTimeout", timeout_ms)
    provider_options.setdefault("chunkTimeout", timeout_ms)
    env["OPENCODE_CONFIG_CONTENT"] = json.dumps(config, separators=(",", ":"))
    return {**args, "env": env}


def _context_args(
    harness: str, args: dict[str, Any], context_window: int | None
) -> dict[str, Any]:
    """Apply shared context and output settings through each CLI's native options."""
    # The CLI's presented model identity may differ from Inspect's served model.
    args = dict(args)
    model = get_model()
    info = get_model_info(model)
    context = context_window or (info.context_length if info else None)
    # An explicit Model keeps its base config separate from eval/CLI overrides.
    output = model.config.merge(active_generate_config()).max_tokens
    opencode_output = (
        output if output is not None else (info.output_tokens if info else None)
    )
    env = dict(args.get("env") or {})
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
    elif harness == "opencode":
        opencode_model = args.setdefault(
            "opencode_model", args.get("model") or str(model)
        )
        if not isinstance(opencode_model, str) or "/" not in opencode_model:
            raise ValueError("opencode_model must use the provider/model format")
        provider, model_id = opencode_model.split("/", 1)
        if not provider or not model_id:
            raise ValueError("opencode_model must use the provider/model format")
        config = json.loads(env.get("OPENCODE_CONFIG_CONTENT", "{}"))
        if not isinstance(config, dict):
            raise ValueError("OPENCODE_CONFIG_CONTENT must decode to a JSON object")
        config.setdefault("small_model", opencode_model)
        _configure_opencode_compaction(config, context)
        provider_config = config.setdefault("provider", {}).setdefault(provider, {})
        # OpenCode authenticates to the local Inspect bridge; provider credentials stay on the host.
        provider_options = provider_config.setdefault("options", {})
        provider_options.setdefault("apiKey", "sk-none")
        if provider == "openrouter":
            # The bridge sends streamed usage only when the client explicitly requests it.
            provider_options.setdefault("compatibility", "strict")
        if context:
            limit = (
                provider_config.setdefault("models", {})
                .setdefault(model_id, {})
                .setdefault("limit", {})
            )
            limit.setdefault("context", _opencode_context_limit(context))
            if opencode_output is not None:
                limit.setdefault("output", opencode_output)
            if "output" not in limit:
                raise ValueError(
                    "OpenCode context configuration requires max_tokens or model metadata output_tokens"
                )
        env["OPENCODE_CONFIG_CONTENT"] = json.dumps(config)
    elif harness == "gemini_cli" and context_window is not None:
        raise ValueError(
            "Gemini CLI uses its native model context limit; leave context_window null and configure gemini_model in agent_args"
        )
    if env:
        args["env"] = env
    return args


def _configure_opencode_compaction(config: dict[str, Any], context: int | None) -> None:
    """Enable native OpenCode compaction and bounded tool output when context is known."""
    if context is None:
        return
    compaction = config.setdefault("compaction", {})
    if not isinstance(compaction, dict):
        raise ValueError("OPENCODE_CONFIG_CONTENT.compaction must be a JSON object")
    compaction.setdefault("auto", True)
    compaction.setdefault("prune", True)
    compaction.setdefault("tail_turns", OPENCODE_TAIL_TURNS)
    compaction.setdefault("preserve_recent_tokens", _opencode_preserve_tokens(context))
    compaction.setdefault("reserved", _opencode_reserved_tokens(context))

    tool_output = config.setdefault("tool_output", {})
    if not isinstance(tool_output, dict):
        raise ValueError("OPENCODE_CONFIG_CONTENT.tool_output must be a JSON object")
    tool_output.setdefault("max_lines", OPENCODE_TOOL_OUTPUT_MAX_LINES)
    tool_output.setdefault("max_bytes", OPENCODE_TOOL_OUTPUT_MAX_BYTES)


def _opencode_context_limit(context: int) -> int:
    """Advertise a conservative model context so OpenCode compacts before the provider limit."""
    return max(1, context // OPENCODE_CONTEXT_FRACTION)


def _opencode_reserved_tokens(context: int) -> int:
    """Reserve request headroom inside the conservative OpenCode context limit."""
    advertised = _opencode_context_limit(context)
    return min(
        max(OPENCODE_MIN_RESERVED_TOKENS, context // 4),
        max(1, advertised // 2),
    )


def _opencode_preserve_tokens(context: int) -> int:
    """Keep a bounded recent tail after each OpenCode compaction."""
    advertised = _opencode_context_limit(context)
    return min(
        max(OPENCODE_MIN_PRESERVE_RECENT_TOKENS, context // 8),
        max(1, advertised // 4),
    )


def _ensure_host_npm_on_path() -> None:
    """Expose nodejs-wheel's npm when the host virtualenv is not activated."""
    if shutil.which("npm"):
        return
    candidates = [Path(sys.executable).resolve().parent]
    try:
        import nodejs_wheel
    except ModuleNotFoundError:
        pass
    else:
        candidates.extend(
            parent / "bin"
            for parent in Path(nodejs_wheel.__file__).resolve().parent.parents
        )
    original_path = os.environ.get("PATH", "")
    for candidate in candidates:
        if (candidate / "npm").exists():
            os.environ["PATH"] = f"{candidate}{os.pathsep}{original_path}"
            if shutil.which("npm"):
                return
    os.environ["PATH"] = original_path


def _opencode_prompt_exceeded_context(harness: str, error: RuntimeError) -> bool:
    """Recognize a complete OpenCode context error without guessing from transcript fragments."""
    prefix = "Error executing opencode agent 1: "
    if harness != "opencode" or not str(error).startswith(prefix):
        return False
    try:
        event = json.loads(str(error).removeprefix(prefix))
    except ValueError:
        return False
    return (
        isinstance(event, dict)
        and event.get("type") == "error"
        and isinstance(event.get("error"), dict)
        and event["error"].get("name") == "ContextOverflowError"
    )
