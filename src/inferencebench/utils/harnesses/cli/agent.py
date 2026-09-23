import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from inspect_ai.agent import Agent, AgentState, BridgedToolsSpec
from inspect_ai.model import ChatMessageUser, get_model
from inspect_ai.model._generate_config import active_generate_config

from .bridge import bridge_poll_timeout
from .options import (
    CLI_HARNESSES,
    _context_args,
    _gemini_timeout_args,
    _opencode_timeout_args,
)


def cli_factory(
    harness: str, args: dict[str, Any], *, package: str
) -> Callable[..., Agent]:
    """Resolve a native CLI factory and validate the caller's configurable arguments."""
    if harness not in CLI_HARNESSES:
        raise ValueError(
            f"Unknown CLI harness {harness!r}; choose from {CLI_HARNESSES}"
        )
    try:
        import inspect_swe
    except ModuleNotFoundError as error:
        if error.name != "inspect_swe":
            raise
        raise ImportError(
            f"Install CLI support with: pip install '{package}[cli]'"
        ) from error
    factory: Callable[..., Agent] = getattr(inspect_swe, harness)
    reserved = {"user", "cwd", "sandbox", "bridged_tools", "filter"}.intersection(args)
    if reserved:
        raise ValueError(
            f"The benchmark supplies these harness arguments: {sorted(reserved)}"
        )
    unknown = args.keys() - inspect.signature(factory).parameters.keys()
    if unknown:
        raise TypeError(f"Unknown {harness} arguments: {sorted(unknown)}")
    return factory


async def run_cli(
    state: AgentState,
    *,
    harness: str,
    factory: Callable[..., Agent],
    args: dict[str, Any],
    user: str,
    cwd: str,
    sandbox: str,
    bridged_tools: list[BridgedToolsSpec],
    model_filter: Any,
    context_window: int | None,
    cli_poll_timeout: int | None,
    bridge_poll_seconds: int | None,
    scoring_only: bool,
    continuation: Callable[[], str | bool],
    before_turn: Callable[[AgentState], Awaitable[None]] | None,
    on_state: Callable[[AgentState], None] | None,
    on_error: Callable[[RuntimeError], bool] | None,
) -> AgentState:
    """Run one native CLI session with shared context, timeouts, and continuation handling."""
    options = _context_args(harness, args, context_window)
    if harness in ("gemini_cli", "opencode"):
        config = get_model().config.merge(active_generate_config())
        if harness == "gemini_cli":
            options = await _gemini_timeout_args(options, config.attempt_timeout)
        else:
            options = _opencode_timeout_args(options, config.attempt_timeout)
    cli = factory(
        user=user,
        cwd=cwd,
        sandbox=sandbox,
        bridged_tools=bridged_tools,
        **dict(options, filter=model_filter),
    )
    while True:
        if before_turn is not None:
            await before_turn(state)
        try:
            with bridge_poll_timeout(
                bridge_poll_seconds if harness == "opencode" else None,
                cli_timeout=cli_poll_timeout,
            ):
                state = await cli(state)
                if on_state is not None:
                    on_state(state)
        except RuntimeError as error:
            if on_error is not None and on_error(error):
                return state
            raise
        if scoring_only:
            return state
        note = continuation()
        if note is False:
            return state
        state.messages.append(ChatMessageUser(content=str(note)))
