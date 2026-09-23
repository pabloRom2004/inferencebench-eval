from collections.abc import Sequence
from typing import Any

from inspect_ai.agent import Agent, AgentContinue, react
from inspect_ai.model import CompactionSummary
from inspect_ai.tool import Tool, ToolDef, ToolSource


def react_agent(
    *,
    tools: Sequence[Tool | ToolDef | ToolSource],
    prompt: str | None,
    submit: bool,
    on_continue: AgentContinue,
    context_window: int | None,
    compaction_threshold: float | None,
    **options: Any,
) -> Agent:
    """Construct native ReAct with explicit task tools, continuation, and compaction settings."""
    threshold = (
        int(context_window * compaction_threshold)
        if context_window and compaction_threshold
        else compaction_threshold
    )
    return react(
        prompt=prompt,
        tools=tools,
        submit=submit,
        on_continue=on_continue,
        compaction=CompactionSummary(threshold=threshold)
        if threshold is not None
        else None,
        **options,
    )
