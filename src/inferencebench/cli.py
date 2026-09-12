"""Inspect SWE adapters with shared continuation, tools, and model context settings."""

import inspect
import json
from typing import Any

import inspect_swe
from inspect_ai.agent import Agent, AgentState, BridgedToolsSpec, agent
from inspect_ai.model import ChatMessageUser, GenerateInput, get_model, get_model_info
from inspect_ai.model._generate_config import active_generate_config
from inspect_ai.util import sandbox

from inferencebench.reminders import cli_reminders, nudge, with_deadline
from inferencebench.run_config import load_config
from inferencebench.tools import web_search

DEFAULT_AGENT_ARGS = load_config()["solver"]["args"]
CLI_HARNESSES = ("claude_code", "codex_cli", "gemini_cli", "kimi_code", "opencode")


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
) -> Agent:
    """Resume the same native CLI session after early exits, using Inspect for all model calls."""
    if harness not in CLI_HARNESSES:
        raise ValueError(
            f"Unknown CLI harness {harness!r}; choose from {CLI_HARNESSES}"
        )
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
            state = await cli(state)
            continuation = nudge(nudge_prompt)
            if continuation is False:
                return state
            state.messages.append(ChatMessageUser(content=str(continuation)))

    return with_deadline(execute)


def context_args(harness: str, args: dict[str, Any]) -> dict[str, Any]:
    """Pass the served model's context and output limits to each CLI's native configuration."""
    args = dict(args)
    model = get_model()
    info = get_model_info(model)
    context = info.context_length if info else None
    output = model.config.merge(active_generate_config()).max_tokens
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
    elif harness == "opencode" and context:
        identity = args.setdefault("opencode_model", args.get("model") or str(model))
        provider, model_id = identity.split("/", 1)
        config = json.loads(env.get("OPENCODE_CONFIG_CONTENT", "{}"))
        config.setdefault("small_model", identity)
        provider_config = config.setdefault("provider", {}).setdefault(provider, {})
        options = provider_config.setdefault("options", {})
        options.setdefault("apiKey", "sk-none")
        if provider == "openrouter":
            options.setdefault("compatibility", "strict")
        limits = (
            provider_config.setdefault("models", {})
            .setdefault(model_id, {})
            .setdefault("limit", {})
        )
        limits.setdefault("context", context // 2)
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
