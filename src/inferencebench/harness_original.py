import time

import inspect_swe
from inspect_ai.agent import Agent, AgentState, agent, as_solver
from inspect_ai.model import (
    ChatMessage,
    ChatMessageUser,
    GenerateConfig,
    GenerateInput,
    Model,
)
from inspect_ai.solver import Solver, solver
from inspect_ai.tool import ToolChoice, ToolInfo
from inspect_ai.util import current_checkpointer, store

from inferencebench.harness_default import file_cli_prompts, with_deadline
from inferencebench.prompts import CONTINUE_PROMPT, NUDGE_PROMPT, ORIGINAL_CLI_CONTEXT
from inferencebench.utils.harnesses.cli.checkpoints import (
    BRIDGE_CHECKPOINT_HARNESSES,
    checkpointed_cli,
)
from inferencebench.utils.reminders import summary_request
from inferencebench.utils.run_config import load_config

ORIGINAL_AGENT_ARGS = load_config("run_configs/original.yaml")["solver"]["args"]


@solver
def original_agent(
    harness: str = ORIGINAL_AGENT_ARGS["harness"],
    version: str = ORIGINAL_AGENT_ARGS["version"],
    continue_until_deadline: bool = ORIGINAL_AGENT_ARGS["continue_until_deadline"],
    env: dict[str, str] = ORIGINAL_AGENT_ARGS["env"],
) -> Solver:
    """Run the original coding CLI and resume early exits until the upstream two-hour budget expires."""
    cli = as_solver(original_cli(harness, version, continue_until_deadline, env))

    async def solve(state, generate):
        """Add upstream's non-interactive note, then run the checkpoint-aware CLI session."""
        if harness == "claude_code":
            state.messages.append(ChatMessageUser(content=ORIGINAL_CLI_CONTEXT.prompt))
        return await cli(state, generate)

    return with_deadline(solve)


@agent
def original_cli(
    harness: str,
    version: str,
    continue_until_deadline: bool,
    env: dict[str, str],
) -> Agent:
    """Keep one native CLI session and one checkpoint owner across continuations, restores, and scoring."""
    args = {"version": version, "cwd": "/home/agent/task", "user": "root", "env": env}
    if harness == "claude_code":
        args.update(permission_mode="bypassPermissions", retry_refusals=0)
    if harness in BRIDGE_CHECKPOINT_HARNESSES:
        args["filter"] = checkpoint_filter
    cli = getattr(inspect_swe, harness)(**args)

    async def run_cli(state: AgentState, *, scoring_only: bool = False) -> AgentState:
        """Resume the same CLI session with the remaining budget after each early exit."""
        while True:
            if harness == "claude_code":
                await file_cli_prompts(state)
            state = await cli(state)
            # A checkpoint restore can extend the deadline, so read it after every exit.
            end = store().get("deadline")
            remaining = int(end - time.time()) if end is not None else None
            if (
                scoring_only
                or not continue_until_deadline
                or (remaining is not None and remaining <= 0)
            ):
                return state
            state.messages.append(
                ChatMessageUser(
                    content=(
                        CONTINUE_PROMPT.prompt.format(
                            minutes=remaining // 60, seconds=remaining
                        )
                        if remaining is not None
                        else NUDGE_PROMPT.prompt
                    )
                )
            )

    return checkpointed_cli(
        harness,
        run_cli,
        checkpoint_prefix="inferencebench_original",
        continue_completed=lambda: False,
        continuation=lambda: False,
    )


async def checkpoint_filter(
    model: Model,
    messages: list[ChatMessage],
    tools: list[ToolInfo],
    tool_choice: ToolChoice | None,
    config: GenerateConfig,
) -> None:
    """Offer a checkpoint at task requests for CLI adapters that do not tick one themselves, leaving the request unchanged."""
    cp = current_checkpointer()
    request = GenerateInput(
        input=messages, tools=tools, tool_choice=tool_choice, config=config
    )
    if cp is not None and tools and not summary_request(request):
        await cp.tick()
