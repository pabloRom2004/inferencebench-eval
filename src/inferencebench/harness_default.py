import asyncio
import time
from functools import partial

from inspect_ai.agent import AgentState, as_solver, react
from inspect_ai.model import ChatMessageUser, CompactionSummary
from inspect_ai.solver import Solver, solver
from inspect_ai.tool import bash, python
from inspect_ai.util import sample_limits, store

from inferencebench.prompts import NUDGE_PROMPT, TOKEN_BUDGET_REMINDER
from inferencebench.run_config import load_config
from inferencebench.tools import web_search

DEFAULT_AGENT_ARGS = load_config()["solver"]["args"]


def budget_reminder() -> str:
    """Describe the effective Inspect token budget, including runtime overrides and consumed tokens."""
    budget = sample_limits().token
    if budget.limit is None:
        return ""
    return TOKEN_BUDGET_REMINDER.prompt.format(
        used=budget.usage,
        limit=budget.limit,
        remaining=max(0, budget.remaining),
    )


@solver
def react_agent(
    tools: list[str] = DEFAULT_AGENT_ARGS["tools"],
    web_search_args: dict = DEFAULT_AGENT_ARGS["web_search_args"],
    compaction_threshold: float = DEFAULT_AGENT_ARGS["compaction_threshold"],
    submit: bool = DEFAULT_AGENT_ARGS["submit"],
    nudge_prompt: bool = DEFAULT_AGENT_ARGS["nudge_prompt"],
    token_budget_reminder: bool = DEFAULT_AGENT_ARGS["token_budget_reminder"],
) -> Solver:
    """Run ReAct with optional submission, continuation nudges, and live token-budget reminders."""
    available = {
        "bash": bash,
        "python": python,
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
            notes.append(NUDGE_PROMPT.prompt)
        if token_budget_reminder:
            notes.append(budget_reminder())
        return "\n\n".join(note for note in notes if note) or True

    agent = as_solver(
        react(
            prompt=None,
            tools=[available[name]() for name in tools],
            submit=submit,
            on_continue=on_continue,
            compaction=CompactionSummary(threshold=compaction_threshold),
        )
    )

    async def solve(state, generate):
        """Announce the token budget and apply a wall-clock deadline only when explicitly configured."""
        if token_budget_reminder and (reminder := budget_reminder()):
            state.messages.append(ChatMessageUser(content=reminder))

        # The maintained default has no deadline; original settings can retain one.
        end = store().get("deadline")
        deadline = asyncio.timeout(
            max(0, end - time.time()) if end is not None else None
        )
        try:
            async with deadline:
                return await agent(state, generate)
        except TimeoutError:
            if not deadline.expired():
                raise
            state.metadata["agent_deadline_reached"] = True
            return state

    return solve
