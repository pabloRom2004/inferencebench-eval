import asyncio
import functools
import importlib
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from inspect_ai import eval_set
from inspect_ai.agent import as_solver
from inspect_ai.agent._bridge.types import AgentBridge
from inspect_ai.log import read_eval_log
from inspect_ai.model import (
    ChatMessageTool,
    GenerateConfig,
    ModelOutput,
    ModelUsage,
    get_model,
)
from inspect_ai.scorer import scorer
from inspect_ai.tool import ToolInfo
from inspect_ai.util import (
    CheckpointConfig,
    TurnInterval,
    current_checkpointer,
    sample_limits,
    store,
)

from inferencebench import original_agent, react_agent
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
        retry_cleanup=False,
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
    # A provider failure resumes from the checkpoint without grading the partial submission.
    assert env.terminate.await_count == 1
    if failure == "model":
        previous = [read_eval_log(path) for path in (tmp_path / "evals").glob("*.eval")]
        failed = next(log.samples[0] for log in previous if log.samples[0].error)
        assert not failed.scores


@pytest.mark.parametrize("harness", ["react", "original"])
def test_restore_extends_deadline_and_keeps_usage(
    local_task, monkeypatch, tmp_path, harness
):
    """Resume ReAct and the original CLI after a provider failure with cumulative usage and the lost time returned."""
    task, env = local_task
    environment = importlib.import_module("inferencebench.environment")
    monkeypatch.setattr(environment, "gpu_environment", lambda: env)
    task.dataset[0].metadata["agent_seconds"] = 3600
    generations = 0
    restored = []

    def output(messages, tools, tool_choice, config):
        """Save one tool turn, fail the next request once, then finish from the restored conversation."""
        nonlocal generations
        generations += 1
        if generations == 2:
            raise RuntimeError("injected provider failure after saved progress")
        if generations == 3:
            restored.append(([m.role for m in messages], sample_limits().token.usage))
        result = (
            ModelOutput.for_tool_call(
                "mockllm/subject", "bash", {"cmd": "echo saved-turn"}
            )
            if generations == 1
            else ModelOutput.from_content("mockllm/subject", "Finished.")
        )
        result.usage = ModelUsage(input_tokens=100, output_tokens=100, total_tokens=200)
        return result

    if harness == "react":
        task.solver = react_agent(nudge_prompt=False, token_budget_reminder=False)
    else:

        @functools.wraps(importlib.import_module("inspect_swe").claude_code)
        def native_cli(**kwargs):
            """Keep the native checkpoint and bridge contract while replacing the CLI process."""

            async def execute(state):
                """Generate through the native bridge's checkpoint session until the model stops calling tools."""
                async with importlib.import_module(
                    "inspect_swe._claude_code.claude_code"
                ).checkpointer() as cp:
                    bridge = AgentBridge(state, checkpointer=cp)
                    if cp.attempt == "resume_for_scoring":
                        return bridge.state
                    while True:
                        await cp.tick()
                        state.output = await get_model().generate(state.messages)
                        state.messages = [*state.messages, state.output.message]
                        if not state.output.message.tool_calls:
                            return state
                        state.messages.append(
                            ChatMessageTool(
                                content="saved-turn",
                                tool_call_id=state.output.message.tool_calls[0].id,
                            )
                        )

            return execute

        monkeypatch.setattr(
            importlib.import_module("inspect_swe"), "claude_code", native_cli
        )
        monkeypatch.setattr(CLI, "sandbox", lambda name=None: env)
        env.write_file = AsyncMock()
        task.solver = original_agent(continue_until_deadline=False)
    task.fail_on_error = True
    success, logs = eval_set(
        task,
        model=get_model("mockllm/subject", custom_outputs=output, memoize=False),
        model_roles={"integrity": judge_model()},
        checkpoint=CheckpointConfig(
            trigger=TurnInterval(every=1),
            checkpoints_location=str(tmp_path / "checkpoints"),
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
    assert sample.scores["inference_speedup"].value == {"speedup": 2.0}
    [(roles, usage)] = restored
    # The saved tool turn survives; the CLI wrapper then asks its native session to continue.
    assert roles[-2:] == (
        ["assistant", "tool"] if harness == "react" else ["tool", "user"]
    )
    assert usage == 200 and sample.token_limit_usage == 400
    [report] = [
        event.data["report"]["data"]
        for event in sample.events
        if event.event == "info"
        and event.source == "checkpoint"
        and event.data.get("event") == "resume"
    ]
    assert report["downtime_seconds"] > 0
    assert sample.store["deadline"] == pytest.approx(
        report["previous_deadline"] + report["downtime_seconds"]
    )
    timer = [
        call.args[0]
        for call in env.exec.await_args_list
        if "create_timer.sh" in " ".join(call.args[0])
    ]
    restored_at = report["checkpoint_saved_at"] + report["downtime_seconds"]
    assert float(timer[-1][2]) * 3600 == pytest.approx(
        report["deadline"] - restored_at, abs=5
    )


def test_resume_for_scoring_keeps_deadline(monkeypatch):
    """Leave a completed agent's deadline alone and point grading at the fresh preparation."""
    environment = importlib.import_module("inferencebench.environment")
    values = {"deadline": 100.0, "checkpoint_saved_at": 50.0, "artifacts": "old"}
    monkeypatch.setattr(
        environment,
        "store",
        lambda: SimpleNamespace(get=values.get, set=values.__setitem__),
    )
    state = SimpleNamespace(metadata={"artifacts": "fresh"})
    report = asyncio.run(environment.resume_environment(state, "resume_for_scoring"))
    assert values == {
        "deadline": 100.0,
        "checkpoint_saved_at": 50.0,
        "artifacts": "fresh",
    }
    assert "downtime_seconds" not in report.data


async def test_deadline_reschedule_moves_running_agent(monkeypatch):
    """Let a restored agent run past the setup deadline when the restore extends it, then stop at the new one."""
    from inferencebench import harness_default

    end = time.time() + 0.2
    monkeypatch.setattr(
        harness_default, "store", lambda: SimpleNamespace(get=lambda key: end)
    )
    state = SimpleNamespace(metadata={})
    finished = []

    async def restored(state, generate):
        """Extend the deadline the way a checkpoint restore does, then outlast the original one."""
        harness_default.reschedule_deadline(time.time() + 0.6)
        await asyncio.sleep(0.4)
        finished.append(True)
        await asyncio.sleep(10)

    started = time.time()
    assert await harness_default.with_deadline(restored)(state, None) is state
    assert finished and 0.5 < time.time() - started < 2
    assert state.metadata["agent_deadline_reached"]


@pytest.mark.parametrize("alive", [True, False])
async def test_react_fails_on_lost_sandbox(monkeypatch, alive):
    """Fail a ReAct attempt when its sandbox stopped answering, but keep going after a transient tool error."""
    from inspect_ai.agent import AgentState
    from inspect_ai.model import ChatMessageAssistant
    from inspect_ai.tool import ToolCallError
    from inspect_ai.util import SandboxUnavailableError

    from inferencebench import harness_default

    probe = AsyncMock(
        side_effect=None if alive else SandboxUnavailableError("pod deleted")
    )
    monkeypatch.setattr(harness_default, "sandbox", lambda: SimpleNamespace(exec=probe))
    state = AgentState(
        messages=[
            ChatMessageAssistant(content=""),
            ChatMessageTool(
                content="",
                tool_call_id="1",
                error=ToolCallError("sandbox_unavailable", "SSH failed"),
            ),
        ]
    )
    if alive:
        await harness_default.require_live_sandbox(state)
    else:
        with pytest.raises(SandboxUnavailableError):
            await harness_default.require_live_sandbox(state)
    probe.assert_awaited_once()
