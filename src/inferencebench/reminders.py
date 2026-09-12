"""Shared continuation and budget messages for native and CLI agents."""

import asyncio
import time
from functools import wraps

from inspect_ai.agent import Agent
from inspect_ai.model import ChatMessageUser, GenerateInput
from inspect_ai.solver import Solver
from inspect_ai.util import sample_limits, store

from inferencebench.prompts import NUDGE_PROMPT, TOKEN_BUDGET_REMINDER


def budget_reminder() -> str:
    """Describe the effective token budget, including runtime overrides and cached input."""
    budget = sample_limits().token
    if budget.limit is None:
        return ""
    return TOKEN_BUDGET_REMINDER.prompt.format(
        used=budget.usage, limit=budget.limit, remaining=max(0, budget.remaining)
    )


def nudge(enabled: bool) -> str | bool:
    """Continue a voluntary completion while recording the continuation in the sample store."""
    if not enabled:
        return False
    store().set("nudges_used", store().get("nudges_used", 0) + 1)
    return NUDGE_PROMPT.prompt


def cli_reminders(request: GenerateInput, enabled: bool) -> GenerateInput:
    """Refresh CLI budget messages without modifying native compaction or title requests."""
    last_user = next((m.text for m in reversed(request.input) if m.role == "user"), "")
    summary = (
        "Please provide your summary based on the conversation so far" in last_user
        or last_user.startswith(
            "You are about to run out of context. Create a handoff summary"
        )
    )
    if enabled and request.tools and not summary and (note := budget_reminder()):
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
        except TimeoutError:
            if not deadline.expired():
                raise
            state.metadata["agent_deadline_reached"] = True
            return state

    return run
