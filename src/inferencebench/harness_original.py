import asyncio
import time

import inspect_swe
from inspect_ai.agent import as_solver
from inspect_ai.model import ChatMessageUser
from inspect_ai.solver import Solver, solver
from inspect_ai.util import store

from inferencebench.prompts import CONTINUE_PROMPT, NUDGE_PROMPT, ORIGINAL_CLI_CONTEXT
from inferencebench.run_config import load_config

ORIGINAL_AGENT_ARGS = load_config("run_configs/original.yaml")["solver"]["args"]


@solver
def original_agent(
    harness: str = ORIGINAL_AGENT_ARGS["harness"],
    version: str = ORIGINAL_AGENT_ARGS["version"],
    continue_until_deadline: bool = ORIGINAL_AGENT_ARGS["continue_until_deadline"],
    env: dict[str, str] = ORIGINAL_AGENT_ARGS["env"],
) -> Solver:
    """Run the original coding CLI and resume early exits until the upstream two-hour budget expires."""
    factory = getattr(inspect_swe, harness)

    args = {"version": version, "cwd": "/home/agent/task", "user": "root", "env": env}
    if harness == "claude_code":
        args.update(permission_mode="bypassPermissions", retry_refusals=0)

    agent = as_solver(factory(**args))

    async def solve(state, generate):
        """Preserve the CLI session between continuations and enforce its elapsed-time deadline."""
        if harness == "claude_code":
            state.messages.append(ChatMessageUser(content=ORIGINAL_CLI_CONTEXT.prompt))

        end = store().get("deadline")
        deadline = asyncio.timeout(max(0, end - time.time()) if end is not None else None)
        try:
            async with deadline:
                while True:
                    state.completed = False
                    state = await agent(state, generate)
                    remaining = int(end - time.time()) if end is not None else None
                    if not continue_until_deadline or (remaining is not None and remaining <= 0):
                        return state

                    # Resume the same CLI session with the remaining budget.
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
        except TimeoutError:
            if not deadline.expired():
                raise
            state.metadata["agent_deadline_reached"] = True
            return state

    return solve
