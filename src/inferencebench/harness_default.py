import asyncio
import time
from functools import partial, wraps
from typing import Any

from inspect_ai.agent import Agent, AgentState, BridgedToolsSpec, agent, as_solver
from inspect_ai.model import (
    ChatMessage,
    ChatMessageUser,
    GenerateConfig,
    GenerateInput,
    Model,
)
from inspect_ai.solver import Solver, solver
from inspect_ai.tool import ToolChoice, ToolInfo, bash, python
from inspect_ai.util import current_checkpointer, sandbox, store

from inferencebench.prompts import NUDGE_PROMPT, TOKEN_BUDGET_REMINDER
from inferencebench.tools import web_search
from inferencebench.utils.harnesses.cli.agent import cli_factory
from inferencebench.utils.harnesses.cli.agent import run_cli as run_native_cli
from inferencebench.utils.harnesses.cli.artifacts import prepare_cli_archive
from inferencebench.utils.harnesses.cli.checkpoints import (
    BRIDGE_CHECKPOINT_HARNESSES,
    checkpointed_cli,
)
from inferencebench.utils.harnesses.cli.options import (
    CLI_HARNESSES,
)
from inferencebench.utils.harnesses.react import react_agent as create_react_agent
from inferencebench.utils.reminders import summary_request, token_reminder
from inferencebench.utils.run_config import load_config

DEFAULT_AGENT_ARGS = load_config()["solver"]["args"]


@solver
def default_agent(
    harness: str = DEFAULT_AGENT_ARGS["harness"],
    harness_args: dict[str, Any] | None = DEFAULT_AGENT_ARGS["harness_args"],
    tools: list[str] = DEFAULT_AGENT_ARGS["tools"],
    web_search_args: dict = DEFAULT_AGENT_ARGS["web_search_args"],
    tool_timeout: int | None = DEFAULT_AGENT_ARGS["tool_timeout"],
    compaction_threshold: float = DEFAULT_AGENT_ARGS["compaction_threshold"],
    submit: bool = DEFAULT_AGENT_ARGS["submit"],
    nudge_prompt: bool = DEFAULT_AGENT_ARGS["nudge_prompt"],
    token_budget_reminder: bool = DEFAULT_AGENT_ARGS["token_budget_reminder"],
    cli_poll_timeout: int | None = DEFAULT_AGENT_ARGS["cli_poll_timeout"],
) -> Solver:
    """Select the maintained ReAct or CLI harness from the same run configuration."""
    shared = dict(
        web_search_args=web_search_args,
        nudge_prompt=nudge_prompt,
        token_budget_reminder=token_budget_reminder,
    )
    if harness == "react":
        return react_agent(**({
            **shared,
            "tools": tools,
            "tool_timeout": tool_timeout,
            "compaction_threshold": compaction_threshold,
            "submit": submit,
        } | dict(harness_args or {})))
    return as_solver(cli_agent(
        harness,
        harness_args,
        cli_poll_timeout=cli_poll_timeout,
        **shared,
    ))


@solver
def react_agent(
    tools: list[str] = DEFAULT_AGENT_ARGS["tools"],
    web_search_args: dict = DEFAULT_AGENT_ARGS["web_search_args"],
    tool_timeout: int | None = DEFAULT_AGENT_ARGS["tool_timeout"],
    compaction_threshold: float = DEFAULT_AGENT_ARGS["compaction_threshold"],
    submit: bool = DEFAULT_AGENT_ARGS["submit"],
    nudge_prompt: bool = DEFAULT_AGENT_ARGS["nudge_prompt"],
    token_budget_reminder: bool = DEFAULT_AGENT_ARGS["token_budget_reminder"],
) -> Solver:
    """Run ReAct with optional submission, continuation nudges, and live token-budget reminders."""
    if tool_timeout is not None and (
        type(tool_timeout) is not int or tool_timeout <= 0
    ):
        raise ValueError("tool_timeout must be a positive integer or null")
    # Model API timeouts do not bound tool execution; the shell and Python tools get their own limit.
    available = {
        "bash": partial(bash, timeout=tool_timeout),
        "python": partial(python, timeout=tool_timeout),
        "web_search": partial(web_search, **web_search_args),
    }

    async def on_continue(state: AgentState) -> bool | str:
        """Keep ordinary tool turns running and nudge early answers when enabled."""
        has_tools = bool(state.output.message.tool_calls)
        if not has_tools and not nudge_prompt:
            return False

        # Native Inspect limits end the attempt; nudges have no count limit.
        notes = []
        if not has_tools:
            notes.append(str(nudge(nudge_prompt)))
        if token_budget_reminder:
            notes.append(budget_reminder())
        return "\n\n".join(note for note in notes if note) or True

    agent = as_solver(
        create_react_agent(
            prompt=None,
            tools=[available[name]() for name in tools],
            submit=submit,
            on_continue=on_continue,
            compaction_threshold=compaction_threshold,
            context_window=None,
        )
    )

    async def solve(state, generate):
        """Announce the token budget and apply a wall-clock deadline only when explicitly configured."""
        if token_budget_reminder and (reminder := budget_reminder()):
            state.messages.append(ChatMessageUser(content=reminder))

        return await agent(state, generate)

    return with_deadline(solve)


def budget_reminder() -> str:
    """Describe the effective token budget, including runtime overrides and cached input."""
    return token_reminder(TOKEN_BUDGET_REMINDER.prompt)


def nudge(enabled: bool) -> str | bool:
    """Continue a voluntary completion while recording the continuation in the sample store."""
    if not enabled:
        return False
    store().set("nudges_used", store().get("nudges_used", 0) + 1)
    return NUDGE_PROMPT.prompt


def cli_reminders(request: GenerateInput, enabled: bool) -> GenerateInput:
    """Refresh CLI budget messages without modifying native compaction or title requests."""
    if (
        enabled
        and request.tools
        and not summary_request(request)
        and (note := budget_reminder())
    ):
        return GenerateInput(
            input=[*request.input, ChatMessageUser(content=note)],
            tools=request.tools,
            tool_choice=request.tool_choice,
            config=request.config,
        )
    return request


def with_deadline(solve: Solver | Agent):
    """Apply the optional optimization deadline while preserving unrelated timeout errors."""

    @wraps(solve)
    async def run(state, *args):
        """Stop at the setup-owned deadline and leave the saved server available for grading."""
        end = store().get("deadline")
        deadline = asyncio.timeout(
            max(0, end - time.time()) if end is not None else None
        )
        try:
            async with deadline:
                return await solve(state, *args)
        except Exception:
            # A transfer or tool cut off by the deadline can surface as another error type.
            if not deadline.expired():
                raise
            state.metadata["agent_deadline_reached"] = True
            return state

    return run


async def file_cli_prompts(state: AgentState) -> None:
    """Keep pending user instructions out of argv so server-oriented pkill cannot match them."""
    for message in reversed(state.messages):
        if message.role == "assistant":
            break
        if message.role == "user" and not message.text.startswith(
            "Read /tmp/inferencebench-input-"
        ):
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
    """Supply task policy to the checkpoint-aware native CLI runtime."""
    if harness not in CLI_HARNESSES:
        raise ValueError(
            f"Unknown CLI harness {harness!r}; choose from {CLI_HARNESSES}"
        )
    if cli_poll_timeout is not None and (
        type(cli_poll_timeout) is not int or cli_poll_timeout <= 0
    ):
        raise ValueError("cli_poll_timeout must be a positive integer or null")
    args = dict(harness_args or {})
    factory = cli_factory(harness, args, package="inferencebench")
    # Native multi-attempt agents invoke the destructive final scorer between attempts.
    if args.get("attempts", 1) != 1:
        raise ValueError(
            "CLI attempts must be 1; continuation resumes the existing session"
        )
    if harness == "claude_code":
        args["disallowed_tools"] = list(
            dict.fromkeys([*(args.get("disallowed_tools") or []), "WebSearch"])
        )

    async def model_filter(
        model: Model,
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice | None,
        config: GenerateConfig,
    ) -> GenerateInput:
        """Checkpoint task requests and refresh budgets without changing native summaries."""
        request = GenerateInput(
            input=messages, tools=tools, tool_choice=tool_choice, config=config
        )
        cp = current_checkpointer()
        if (
            harness in BRIDGE_CHECKPOINT_HARNESSES
            and cp is not None
            and tools
            and not summary_request(request)
        ):
            await cp.tick()
        return cli_reminders(request, token_budget_reminder)

    async def run_cli(state: AgentState, *, scoring_only: bool = False) -> AgentState:
        """Retain one native session while supplying task tools and workspace environment."""
        await prepare_cli_archive(harness, args.get("version"))
        return await run_native_cli(
            state,
            harness=harness,
            factory=factory,
            args={
                **args,
                "env": {
                    **(store().get("workspace_env") or {}),
                    **(args.get("env") or {}),
                },
            },
            cwd="/home/agent/task",
            user="root",
            sandbox="default",
            bridged_tools=[
                BridgedToolsSpec(
                    name="inferencebench", tools=[web_search(**web_search_args)]
                )
            ],
            model_filter=model_filter,
            context_window=None,
            cli_poll_timeout=cli_poll_timeout,
            bridge_poll_seconds=cli_poll_timeout,
            scoring_only=scoring_only,
            continuation=lambda: nudge(nudge_prompt),
            before_turn=file_cli_prompts if harness == "claude_code" else None,
            on_state=None,
            on_error=None,
        )

    return with_deadline(
        checkpointed_cli(
            harness,
            run_cli,
            checkpoint_prefix="inferencebench_cli",
            continue_completed=lambda: (
                nudge_prompt and store().get("exit_reason") is None
            ),
            continuation=lambda: nudge(True),
        )
    )
