import importlib
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager, contextmanager
from contextvars import ContextVar
from typing import Any, Literal, TypeVar, cast

from inspect_ai.agent import Agent, AgentState
from inspect_ai.model import ChatMessage, ChatMessageAssistant, ChatMessageUser
from inspect_ai.util import Checkpointer, checkpointer
from inspect_ai.util._checkpoint.report import ResumeReport

T = TypeVar("T")
_SESSION: ContextVar["NativeCheckpoint | None"] = ContextVar(
    "native_cli_checkpoint", default=None
)


class NativeCheckpoint:
    """Keep native bridge registrations alive across voluntary CLI continuations."""

    def __init__(self, checkpoint: Checkpointer, resume_agent: bool = False) -> None:
        """Reuse the sample's checkpoint without owning its completion boundary."""
        self.checkpoint_session = checkpoint
        self.resume_agent = resume_agent
        self.migration_saved = False
        self.callbacks: dict[str, Callable[[], Any]] = {}

    @property
    def attempt(self) -> Literal["initial", "resume", "resume_for_scoring"]:
        """Treat a legacy premature final checkpoint as an agent continuation."""
        return "resume" if self.resume_agent else self.checkpoint_session.attempt

    @property
    def restored(self) -> ResumeReport | None:
        """Expose the unchanged framework resume report."""
        return self.checkpoint_session.restored

    async def tick(self) -> None:
        """Apply the configured save trigger at the native bridge boundary."""
        if self.resume_agent and not self.migration_saved:
            # Replace the old premature completion phase at the first safe
            # boundary, even when its restored trigger is not due yet.
            await self.checkpoint_session.checkpoint()
            self.migration_saved = True
        await self.checkpoint_session.tick()

    async def checkpoint(self) -> None:
        """Commit the current live native session state."""
        await self.checkpoint_session.checkpoint()

    def span_session(self) -> AbstractAsyncContextManager[None]:
        """Delegate explicit span management to the owner."""
        return self.checkpoint_session.span_session()

    def track(
        self,
        key: str,
        callback: Callable[[], T],
        initial_value: T,
        *,
        value_type: type[T] | None = None,
    ) -> T:
        """Register once, then transfer the live value to each replacement bridge."""
        previous = self.callbacks.get(key)
        if previous is not None:
            value = cast(T, previous())
            self.callbacks[key] = callback
            return value
        self.callbacks[key] = callback
        return self.checkpoint_session.track(
            key, lambda: self.callbacks[key](), initial_value, value_type=value_type
        )


@asynccontextmanager
async def native_checkpointer() -> AsyncIterator[Checkpointer]:
    """Borrow the wrapper session, leaving unrelated native agents unchanged."""
    session = _SESSION.get()
    if session is None:
        async with checkpointer() as checkpoint:
            yield checkpoint
    else:
        yield session


@contextmanager
def native_checkpoint_scope(harness: str, session: NativeCheckpoint) -> Iterator[None]:
    """Route only this context's native adapter to its enclosing checkpoint owner."""
    module = importlib.import_module(f"inspect_swe._{harness}.{harness}")
    # Inspect SWE 0.2.71 still opens a sample checkpoint on every invocation.
    # The context-local dispatcher avoids duplicate registrations and premature
    # agent_complete saves without changing other samples' checkpoint behavior.
    setattr(module, "checkpointer", native_checkpointer)
    token = _SESSION.set(session)
    try:
        yield
    finally:
        _SESSION.reset(token)


BRIDGE_CHECKPOINT_HARNESSES = ("gemini_cli", "kimi_code", "opencode")


def checkpointed_cli(
    harness: str,
    run_cli: Callable[..., Awaitable[AgentState]],
    *,
    checkpoint_prefix: str,
    continue_completed: Callable[[], bool],
    continuation: Callable[[], str | bool],
) -> Agent:
    """Preserve one checkpoint owner across native CLI exits, retries, and scoring."""

    async def execute(state: AgentState) -> AgentState:
        """Run checkpoint-aware wrappers for CLI adapters that do not provide one."""
        if harness not in BRIDGE_CHECKPOINT_HARNESSES:
            async with checkpointer() as cp:
                completed = cp.track(
                    f"{checkpoint_prefix}_complete", lambda: completed, False
                )
                state.messages = cp.track(
                    f"{checkpoint_prefix}_messages",
                    lambda: state.messages,
                    state.messages,
                    value_type=list[ChatMessage],
                )
                state.output = cp.track(
                    f"{checkpoint_prefix}_output", lambda: state.output, state.output
                )
                if completed:
                    return state
                if (
                    cp.attempt == "resume"
                    and state.messages
                    and state.messages[-1].role != "user"
                ):
                    state.messages.append(
                        ChatMessageUser(content="Continue from the restored session.")
                    )
                # Older wrappers finalized at a voluntary CLI exit before
                # appending their nudge. Resume that preserved native session.
                legacy_continuation = (
                    cp.attempt == "resume_for_scoring" and continue_completed()
                )
                session = NativeCheckpoint(cp, resume_agent=legacy_continuation)
                if legacy_continuation:
                    state.messages = session.track(
                        "bridge_messages",
                        lambda: state.messages,
                        state.messages,
                        value_type=list[ChatMessage],
                    )
                    state.output = session.track(
                        "bridge_output", lambda: state.output, state.output
                    )
                    state.messages.append(ChatMessageUser(content=str(continuation())))
                with native_checkpoint_scope(harness, session):
                    state = await run_cli(
                        state, scoring_only=session.attempt == "resume_for_scoring"
                    )
                completed = True
                if legacy_continuation:
                    # The old framework phase suppresses automatic finalization.
                    # This marker permits a later scoring-only retry regardless.
                    await cp.checkpoint()
            return state

        async with checkpointer() as cp:
            state.messages = cp.track(
                f"{checkpoint_prefix}_messages",
                lambda: state.messages,
                state.messages,
                value_type=list[ChatMessage],
            )
            state.output = cp.track(
                f"{checkpoint_prefix}_output", lambda: state.output, state.output
            )
            if cp.attempt == "resume_for_scoring":
                return state
            if cp.attempt == "resume":
                # Native adapters require an assistant turn to select their
                # existing session, followed by a user turn to restart it.
                if not any(message.role == "assistant" for message in state.messages):
                    state.messages.append(
                        ChatMessageAssistant(
                            content="The prior native CLI session was restored."
                        )
                    )
                state.messages.append(
                    ChatMessageUser(content="Continue from the restored session.")
                )
            state = await run_cli(state)
            return state

    return execute
