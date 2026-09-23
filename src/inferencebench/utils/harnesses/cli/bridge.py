from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from typing import cast

from inspect_ai.util import (
    ExecRemoteAwaitableOptions,
    ExecRemoteProcess,
    ExecRemoteStreamingOptions,
    ExecResult,
    sandbox,
)
from inspect_ai.util._sandbox._cli import SANDBOX_CLI


@contextmanager
def bridge_poll_timeout(
    timeout: int | None, *, cli_timeout: int | None = None
) -> Iterator[None]:
    """Configure remote-process RPC timeouts on only the current sample's sandbox."""
    if timeout is None and cli_timeout is None:
        yield
        return
    environment = sandbox("default")
    original = environment.exec_remote
    instance_override = "exec_remote" in vars(environment)
    call = cast(Callable[..., Awaitable[ExecRemoteProcess | ExecResult[str]]], original)

    async def exec_remote(
        cmd: list[str],
        options: ExecRemoteStreamingOptions | ExecRemoteAwaitableOptions | None = None,
        *,
        stream: bool = True,
    ) -> ExecRemoteProcess | ExecResult[str]:
        """Adjust the bridge proxy options while preserving other remote commands."""
        if cli_timeout is not None:
            if options is None:
                options = (
                    ExecRemoteStreamingOptions()
                    if stream
                    else ExecRemoteAwaitableOptions()
                )
            if options.poll_timeout is None:
                options = replace(options, poll_timeout=cli_timeout)
        if (
            timeout is not None
            and cmd == [SANDBOX_CLI, "model_proxy"]
            and isinstance(options, ExecRemoteStreamingOptions)
        ):
            options = replace(options, poll_timeout=timeout)
        return await call(cmd, options, stream=stream)

    # Native CLI adapters omit the RPC timeout; the model proxy specifies 600s.
    # Patch the sample-local instance, never the shared framework class.
    setattr(environment, "exec_remote", exec_remote)
    try:
        yield
    finally:
        if instance_override:
            setattr(environment, "exec_remote", original)
        else:
            delattr(environment, "exec_remote")
