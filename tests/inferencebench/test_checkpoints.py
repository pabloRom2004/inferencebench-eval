import functools
import importlib
from unittest.mock import AsyncMock

import pytest
from inspect_ai import eval_set
from inspect_ai.agent import as_solver
from inspect_ai.agent._bridge.types import AgentBridge
from inspect_ai.log import read_eval_log
from inspect_ai.model import GenerateConfig, ModelOutput, ModelUsage, get_model
from inspect_ai.scorer import scorer
from inspect_ai.tool import ToolInfo
from inspect_ai.util import CheckpointConfig, TurnInterval, current_checkpointer, store

from inferencebench.harness_default import cli_agent
from inferencebench.utils.harnesses.cli.options import CLI_HARNESSES
from tests.inferencebench.test_task import fake_judge_cli as fake_judge_cli
from tests.inferencebench.test_task import judge_model
from tests.inferencebench.test_task import local_task as local_task

CLI = importlib.import_module("inferencebench.harness_default")


@pytest.mark.parametrize("harness", CLI_HARNESSES)
@pytest.mark.parametrize("failure", ["model", "scorer"])
def test_cli_checkpoint_preserves_progress_and_scoring(
    local_task, monkeypatch, tmp_path, harness, failure
):
    """Resume every CLI after model or scorer failure without repeating completed work."""
    task, env = local_task
    env.write_file = AsyncMock()
    task.dataset[0].metadata["agent_seconds"] = None
    monkeypatch.setattr(CLI, "sandbox", lambda name=None: env)
    monkeypatch.setattr(
        "inferencebench.utils.harnesses.cli.options.sandbox", lambda name=None: env
    )
    native = importlib.import_module(f"inspect_swe._{harness}.{harness}")
    original_factory = getattr(importlib.import_module("inspect_swe"), harness)
    original_nudge = CLI.nudge
    original_score = task.scorer[0]
    generations = 0
    scorer_calls = 0
    resumes = []

    def output(messages, tools, tool_choice, config):
        """Fail one request after the first saved turn, then return deterministic usage."""
        nonlocal generations
        generations += 1
        if failure == "model" and generations == 2:
            raise RuntimeError("injected provider failure after saved progress")
        result = ModelOutput.from_content("mockllm/subject", "Completed one turn.")
        result.usage = ModelUsage(input_tokens=100, output_tokens=100, total_tokens=200)
        return result

    @functools.wraps(original_factory)
    def native_cli(**kwargs):
        """Retain the real bridge checkpoint contract while replacing native processes."""

        async def turn(state):
            """Filter and generate once, then save the completed task work and accounting."""
            request = await kwargs["filter"](
                get_model(),
                state.messages,
                [ToolInfo(name="Bash", description="Run a shell command")],
                "auto",
                GenerateConfig(),
            )
            state.output = await get_model().generate(request.input)
            state.messages = [*request.input, state.output.message]
            store().set("completed_turns", store().get("completed_turns", 0) + 1)
            await current_checkpointer().checkpoint()
            return state

        async def execute(state):
            """Use native bridge registration for the two adapters that own checkpoints."""
            if harness in ("claude_code", "codex_cli"):
                async with native.checkpointer() as cp:
                    bridge = AgentBridge(state, checkpointer=cp)
                    if cp.attempt == "resume_for_scoring":
                        return bridge.state
                    return await turn(state)
            return await turn(state)

        return execute

    def continue_two_turns(enabled):
        """Stop after two completed turns, counting only continuations that are needed."""
        return (
            original_nudge(enabled) if store().get("completed_turns", 0) < 2 else False
        )

    @scorer(metrics=[])
    def retry_scorer():
        """Exercise the real task scorer after an optional one-time infrastructure error."""

        async def score(state, target):
            """Fail before grading once, so the retry must preserve the completed agent."""
            nonlocal scorer_calls
            scorer_calls += 1
            if failure == "scorer" and scorer_calls == 1:
                raise RuntimeError("injected scorer failure after agent completion")
            return await original_score(state, target)

        return score

    async def resumed(state, attempt):
        """Record the actual resume phase selected from the saved checkpoint."""
        resumes.append(attempt)

    monkeypatch.setattr(importlib.import_module("inspect_swe"), harness, native_cli)
    monkeypatch.setattr(CLI, "nudge", continue_two_turns)
    task.solver = as_solver(cli_agent(harness, cli_poll_timeout=None))
    task.scorer = [retry_scorer()]
    task.on_resume = resumed
    task.fail_on_error = True
    success, logs = eval_set(
        task,
        model=get_model("mockllm/subject", custom_outputs=output, memoize=False),
        model_roles={"integrity": judge_model()},
        checkpoint=CheckpointConfig(
            trigger=TurnInterval(every=1),
            checkpoints_location=str(tmp_path / "checkpoints"),
            retention="retain",
        ),
        log_dir=str(tmp_path / "evals"),
        retry_attempts=2,
        retry_wait=0.001,
        retry_immediate=False,
        display="none",
        log_shared=False,
    )
    assert success
    sample = read_eval_log(logs[-1].location).samples[0]
    assert sample.error is None
    assert resumes == ["resume" if failure == "model" else "resume_for_scoring"]
    assert generations == (3 if failure == "model" else 2)
    assert sample.store["completed_turns"] == 2
    assert sample.token_limit_usage == 400
    assert sample.scores["retry_scorer"].value == {"speedup": 2.0}
    env.terminate.assert_awaited_once()
