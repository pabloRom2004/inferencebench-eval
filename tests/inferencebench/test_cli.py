"""Exercise CLI continuation through the same task and final scorer as ReAct."""

import functools
import importlib

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.agent import agent, as_solver
from inspect_ai.log import read_eval_log
from inspect_ai.model import (
    ChatMessageUser,
    GenerateConfig,
    ModelInfo,
    ModelOutput,
    ModelUsage,
    get_model,
)
from inspect_ai.tool import ToolInfo

from inferencebench.cli import CLI_HARNESSES, cli_agent
from tests.inferencebench.test_task import judge_model
from tests.inferencebench.test_task import local_task as local_task
from tests.inferencebench.test_task import remove_mock_logs as remove_mock_logs

CLI = importlib.import_module("inferencebench.cli")


@pytest.mark.parametrize("harness", CLI_HARNESSES)
@pytest.mark.parametrize("continue_work", [False, True])
def test_cli_continuation_and_scoring(local_task, monkeypatch, harness, continue_work):
    """Keep one CLI session, refresh its budget, and grade once after stopping or reaching the cap."""
    observed = []
    requests = []
    original = getattr(CLI.inspect_swe, harness)

    @functools.wraps(original)
    def factory(**kwargs):
        """Replace process execution while validating the actual native factory arguments."""
        assert callable(original(**kwargs))
        observed.append(kwargs)

        @agent
        def stub():
            """Stand in for a native CLI's requests to the Inspect bridge."""

            async def execute(state):
                """Request one model completion and retain it in the native session state."""
                config = GenerateConfig()
                tools = [ToolInfo(name="Bash", description="Run a shell command")]
                request = await kwargs["filter"](
                    get_model(), state.messages, tools, "auto", config
                )
                requests.append(request)
                output = await get_model().generate(request.input)
                state.messages = [*request.input, output.message]
                return state

            return execute

        return stub()

    monkeypatch.setattr(CLI.inspect_swe, harness, factory)
    monkeypatch.setattr(
        CLI,
        "get_model_info",
        lambda model: ModelInfo(context_length=1048576, output_tokens=131072),
    )
    task, env = local_task
    task.dataset[0].metadata["agent_seconds"] = None
    task.solver = as_solver(cli_agent(harness, nudge_prompt=continue_work))
    outputs = [ModelOutput.from_content("mockllm/subject", "Ready") for _ in range(5)]
    for output in outputs:
        output.usage = ModelUsage(input_tokens=100, output_tokens=100, total_tokens=200)
    [log] = inspect_eval(
        task,
        model=get_model("mockllm/subject", custom_outputs=outputs),
        model_roles={"integrity": judge_model()},
        token_limit=500,
        display="none",
        log_dir="logs",
    )
    assert log.status == "success", log.error
    sample = read_eval_log(log.location).samples[0]
    assert sample.error is None
    assert sample.scores["inference_speedup"].value == {"speedup": 2.0}
    assert len(observed) == 1
    assert observed[0]["cwd"] == "/home/agent/task"
    assert observed[0]["user"] == "root"
    assert "0 / 500" in requests[0].input[-1].text
    if continue_work:
        assert sample.limit.type == "token"
        assert any("200 / 500" in r.input[-1].text for r in requests)
    else:
        assert sample.token_limit_usage == 200
    env.terminate.assert_awaited_once()
    if harness == "claude_code":
        assert observed[0]["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "1048576"


@pytest.mark.parametrize(
    "args", [{"attempts": 2}, {"cwd": "/tmp"}, {"versoin": "auto"}]
)
def test_cli_rejects_destructive_or_unknown_options(args):
    """Reject options that bypass task paths, rescore during optimization, or silently misspell settings."""
    with pytest.raises((ValueError, TypeError)):
        cli_agent("claude_code", args)


def test_cli_summary_keeps_budget_out_of_compaction():
    """Leave summary-only requests untouched without requiring a live sample budget."""
    from inspect_ai.model import GenerateInput

    from inferencebench.reminders import cli_reminders

    summary = GenerateInput(
        input=[
            ChatMessageUser(
                content="Please provide your summary based on the conversation so far"
            )
        ],
        tools=[ToolInfo(name="Bash", description="Run a shell command")],
        tool_choice="auto",
        config=GenerateConfig(),
    )
    assert cli_reminders(summary, True).input == summary.input
