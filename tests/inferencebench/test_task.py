import asyncio
import importlib
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.log import read_eval_log
from inspect_ai.model import ModelOutput, ModelUsage, get_model
from inspect_ai.scorer import SampleScore, Score
from inspect_ai.solver import generate, solver
from inspect_ai.util import ExecResult, store

from inferencebench import default_agent, inference_bench, original_agent, react_agent
from inferencebench.metrics import aggregate_speedup, complete_mean, performance
from inferencebench.prompts import AUTOMATED_TUNING, PROMPTS
from inferencebench.scorers import parse_judgment
from inferencebench.tools import web_search
from inferencebench.utils.run_config import load_config

TASK = importlib.import_module("inferencebench.task")
SCORERS = importlib.import_module("inferencebench.scorers")
HARNESS = importlib.import_module("inferencebench.harness_original")
TOOLS = importlib.import_module("inferencebench.tools")


JUDGE_FILES: dict[str, str] = {}
JUDGE_CALLS: list[dict] = []


@pytest.fixture(autouse=True)
def fake_judge_cli(monkeypatch):
    """Replace Claude Code with a stub that asks the judge model once and writes its verdict lines as upstream's files."""
    from inspect_ai.agent import agent
    from inspect_ai.util import sandbox

    JUDGE_FILES.clear()
    JUDGE_CALLS.clear()

    def claude_code(**kwargs):
        """Record the native arguments and return the stub agent."""
        JUDGE_CALLS.append(kwargs)

        @agent
        def stub():
            """Stand in for Claude Code writing contamination_judgement.txt and disallowed_model_judgement.txt."""

            async def execute(state):
                """Generate once with the judge model and file each verdict line the way the CLI judge does."""
                output = await kwargs["model"].generate(state.messages)
                lines = [line.replace("**", "").replace("`", "").strip() for line in output.completion.splitlines() if line.strip()]
                for name, line in zip(SCORERS.VERDICT_FILES, lines):
                    path = f"{kwargs['cwd']}/{name}"
                    text = line.removeprefix(name + ":").strip() + "\n"
                    try:
                        await sandbox(kwargs["sandbox"]).write_file(path, text)
                    except Exception:
                        JUDGE_FILES[path] = text
                state.messages = [*state.messages, output.message]
                return state

            return execute

        return stub()

    monkeypatch.setattr(SCORERS, "judge_cli", claude_code)


def verdict_file(path):
    """Return a verdict file the fake judge wrote, raising like a sandbox read for anything else."""
    if path not in JUDGE_FILES:
        raise FileNotFoundError(path)
    return JUDGE_FILES[path]


@pytest.fixture(autouse=True)
def remove_mock_logs():
    """Keep temporary mock logs in the flat logs directory and remove only logs created by this test."""
    folder = Path("logs")
    before = set(folder.glob("*.eval"))
    yield
    for path in set(folder.glob("*.eval")) - before:
        if read_eval_log(path, header_only=True).eval.model.startswith("mockllm/"):
            path.unlink()


def measurements(scale=1):
    """Return a successful upstream-shaped measurement with an analytically known speed ratio."""
    return {
        "profiles": {
            "burst": {
                "success_count": 1,
                "ttft": {"p50": 2 / scale},
                "tpot": {"p50": 4 / scale},
                "request_throughput_req_per_s": 3 * scale,
            }
        },
        "quality_check": {"pass": True},
    }


@pytest.fixture
def local_task(monkeypatch, tmp_path):
    """Replace cloud transport with a deterministic final server while retaining the public task and scoring flow."""

    @solver
    def prepare():
        """Construct the trusted local baseline and start the sample's agent deadline."""

        async def solve(state, generate):
            """Record setup execution independently of whichever solver the run selects."""
            seconds = state.metadata["agent_seconds"]
            store().set(
                "deadline", time.time() + seconds if seconds is not None else None
            )
            store().set("artifacts", str(tmp_path))
            store().set("workspace_env", {
                "INFERENCE_BENCH_BASE_MODEL": state.metadata["base_model"],
                "INFERENCE_BENCH_MAX_MODEL_LEN": str(state.metadata["max_model_len"]),
            })
            (tmp_path / "trusted" / "speed").mkdir(parents=True, exist_ok=True)
            (tmp_path / "trusted" / "speed" / "baseline_metrics.json").write_text(json.dumps({"baseline": measurements()}))
            state.metadata["prepared"] = True
            return state

        return solve

    async def read_file(path):
        """Return the fresh final server outputs, the fake judge's verdict files, and launcher or log evidence."""
        if path.endswith("final.json"):
            return json.dumps(measurements(2))
        if path in JUDGE_FILES:
            return JUDGE_FILES[path]
        if path.endswith("_judgement.txt"):
            raise FileNotFoundError(path)
        return "Mistral server log"

    env = SimpleNamespace(
        exec=AsyncMock(
            return_value=ExecResult(success=True, returncode=0, stdout="", stderr="")
        ),
        read_file=read_file,
        terminate=AsyncMock(),
    )
    monkeypatch.setattr(TASK, "prepare_environment", prepare)
    monkeypatch.setattr(SCORERS, "restart_for_scoring", AsyncMock(return_value=env))
    task = inference_bench(scenarios="A", seed_pairs=[[21, 1337]], agent_seconds=2)
    task.sandbox = None
    return task, env


def judge_model(text="no contamination detected\nonly allowed use detected"):
    """Create a credential-free integrity role with a deterministic upstream-format verdict."""
    return get_model(
        "mockllm/judge",
        custom_outputs=[
            ModelOutput.from_content("mockllm/judge", text) for _ in range(6)
        ],
    )


@pytest.mark.parametrize("automated_tuning", [False, True])
def test_tuning_instruction_and_deadline_reach_model(local_task, automated_tuning):
    """Deliver the selected tuning instruction and two-hour budget through the real task and scorer."""
    _, env = local_task
    task = inference_bench(
        agent_seconds=7200,
        automated_tuning=automated_tuning,
        quality_reference_backend="vllm",
        gpu_provider="runpod",
    )
    task.sandbox = None
    task.solver = default_agent(nudge_prompt=False, token_budget_reminder=False)
    requests = []

    def output(messages, tools, tool_choice, config):
        """Observe the actual subject input and finish without making cloud or provider calls."""
        requests.append(messages[0].text)
        return ModelOutput.from_content("mockllm/tuning", "Finished")

    [log] = inspect_eval(
        task,
        model=get_model("mockllm/tuning", custom_outputs=output),
        model_roles={"integrity": judge_model()},
        display="none",
        log_dir="logs",
    )
    assert log.status == "success", log.error
    assert len(requests) == 1
    assert (AUTOMATED_TUNING.prompt in requests[0]) is automated_tuning
    assert "2 hours of wall-clock optimization time" in requests[0]
    assert "no wall-clock optimization limit" not in requests[0]
    assert "{budget_" not in requests[0]
    [sample] = log.samples
    assert sample.metadata["automated_tuning"] is automated_tuning
    assert sample.scores["inference_speedup"].value == {"speedup": 2.0}
    env.terminate.assert_awaited_once()


def test_config_dataset_and_provenance():
    """Select one seed pair per default scenario while retaining all twelve original samples."""
    task = inference_bench(scenarios=None)
    config = load_config()
    assert [sample.id for sample in task.dataset] == [
        f"{scenario}-21-1337" for scenario in "ABCD"
    ]
    assert len(task.dataset) == load_config("eval.yaml")["samples"]
    assert [sample.id for sample in inference_bench(scenarios="B").dataset] == [
        "B-21-1337"
    ]
    assert [sample.id for sample in inference_bench(scenarios=["A", "C"]).dataset] == [
        "A-21-1337",
        "C-21-1337",
    ]
    assert task.dataset[0].metadata["quality_samples"] == 500
    assert task.config.attempt_timeout == config["generate_config"]["attempt_timeout"]
    assert task.token_limit == 100000000
    assert task.dataset[0].metadata["agent_seconds"] == 36000
    assert "10 hours of wall-clock optimization time" in task.dataset[0].input
    assert "Kernel Optimization" in task.dataset[0].input
    assert "{model}" not in task.dataset[0].input
    assert "24cdf88" in PROMPTS["original"].origin
    original = load_config("run_configs/original.yaml")
    assert original["task"]["args"]["agent_seconds"] == 7200
    assert original["task"]["args"]["system_prompt"] == "original"
    assert original["eval_config"]["token_limit"] is None
    original_task = inference_bench(**original["task"]["args"])
    assert original_task.token_limit is None
    assert len(original_task.dataset) == 12
    assert original["task"]["args"]["seed_pairs"] == [
        [21, 1337],
        [248, 428],
        [999, 777],
    ]
    differing_args = {
        "config_defaults",
        "agent_seconds",
        "system_prompt",
        "scorer",
        "seed_pairs",
        "scenarios",
        "request_limit",
        "quality_baseline_max_attempts",
        "quality_reference_backend",
        "seeded_arrivals",
        "scenario_a_output_tokens",
    }
    assert {
        key: value
        for key, value in original["task"]["args"].items()
        if key not in differing_args
    } == {
        key: value
        for key, value in config["task"]["args"].items()
        if key not in differing_args
    }
    assert config["task"]["args"]["scorer"]["args"]["transcript_hint"] is True
    assert original["task"]["args"]["scorer"]["args"] == {
        **config["task"]["args"]["scorer"]["args"],
        "transcript_hint": False,
        "max_grader_attempts": 1,
    }
    assert original["solver"]["args"]["version"] == "2.1.114"


def test_solver_override_and_judge_role(local_task, tmp_path):
    """Keep preparation and scoring when a native solver override and independent mock judge are supplied."""
    task, env = local_task
    [log] = inspect_eval(
        task,
        solver=generate(),
        model="mockllm/subject",
        model_roles={
            "integrity": judge_model(
                "**`contamination_judgement.txt`:** no contamination detected\n**`disallowed_model_judgement.txt`:** only allowed use detected"
            )
        },
        display="none",
        log_dir="logs",
    )
    assert log.status == "success"
    assert log.samples[0].metadata["prepared"]
    assert log.samples[0].scores["inference_speedup"].value == {"speedup": 2.0}
    assert log.results.scores[0].metrics["aggregate_speedup"].value == 2.0
    env.terminate.assert_awaited_once()


def test_react_completes_without_early_exit_loop(local_task, tmp_path):
    """Allow a bounded ReAct smoke run to complete and grade the persisted submission."""
    task, _ = local_task
    task.solver = react_agent(nudge_prompt=False, token_budget_reminder=False)
    subject = get_model(
        "mockllm/subject",
        custom_outputs=[
            ModelOutput.for_tool_call("mockllm/subject", "bash", {}),
            ModelOutput.from_content("mockllm/subject", "Finished"),
        ],
    )
    [log] = inspect_eval(
        task,
        model=subject,
        model_roles={"integrity": judge_model()},
        display="none",
        log_dir="logs",
    )
    assert log.status == "success"
    assert log.samples[0].messages[-1].text == "Finished"
    assert [message.role for message in log.samples[0].messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert log.samples[0].scores["inference_speedup"].value["speedup"] == 2.0


@pytest.mark.parametrize("outcome", ["results", "unavailable", "empty_query"])
def test_react_web_search(local_task, monkeypatch, outcome):
    """Expose web results and recoverable search errors to ReAct before grading the saved server."""
    results = [
        {
            "title": "vLLM tuning",
            "href": "https://docs.vllm.ai/",
            "body": "Batching guidance",
        }
    ]
    search = Mock(return_value=results)
    if outcome == "unavailable":
        search.side_effect = TOOLS.DDGSException("search temporarily unavailable")
    client = Mock(return_value=SimpleNamespace(text=search))
    monkeypatch.setattr(TOOLS, "DDGS", client)

    task, _ = local_task
    task.dataset[0].metadata["agent_seconds"] = None
    task.solver = react_agent(
        nudge_prompt=False,
        token_budget_reminder=False,
        web_search_args={"backend": "duckduckgo", "max_results": 3, "timeout": 7},
    )
    query = "" if outcome == "empty_query" else "vLLM batching"
    subject = get_model(
        "mockllm/subject",
        custom_outputs=[
            ModelOutput.for_tool_call(
                "mockllm/subject", "web_search", {"query": query}
            ),
            ModelOutput.from_content("mockllm/subject", "Ready"),
        ],
    )
    [log] = inspect_eval(
        task,
        model=subject,
        model_roles={"integrity": judge_model()},
        display="none",
        log_dir="logs",
    )
    assert log.status == "success", log.error
    [reply] = [message for message in log.samples[0].messages if message.role == "tool"]
    if outcome == "results":
        assert json.loads(reply.text) == results
    else:
        assert reply.error is not None
        assert (
            "temporarily unavailable" if outcome == "unavailable" else "non-empty"
        ) in reply.error.message
    if outcome == "empty_query":
        client.assert_not_called()
    else:
        client.assert_called_once_with(timeout=7)
        search.assert_called_once_with(query, backend="duckduckgo", max_results=3)
    assert log.samples[0].scores["inference_speedup"].value == {"speedup": 2.0}


def test_search_rejects_unbounded_result_count():
    """Reject a nonpositive result limit before a search request can be sent."""
    with pytest.raises(ValueError, match="positive integers"):
        web_search(backend="duckduckgo", max_results=0, timeout=5)


@pytest.mark.parametrize(
    "harness,continue_until_deadline,seconds",
    [
        ("claude_code", False, 2),
        ("claude_code", True, 2),
        ("claude_code", False, None),
        ("codex_cli", False, None),
        ("opencode", False, None),
        ("gemini_cli", False, None),
    ],
)
def test_original_dispatches_configured_cli(
    local_task, monkeypatch, tmp_path, harness, continue_until_deadline, seconds
):
    """Dispatch the configured original CLI through Inspect while retaining the shared scorer."""
    from inspect_ai.agent import agent

    observed = {}
    real_factory = getattr(HARNESS.inspect_swe, harness)
    version = "2.1.114" if harness == "claude_code" else "auto"

    def factory(**kwargs):
        """Capture the CLI adapter's options and replace remote process execution with a minimal agent."""
        assert callable(real_factory(**kwargs))
        observed.update(kwargs)

        @agent
        def stub():
            """Create a deterministic stand-in for the Inspect SWE bridge boundary."""

            async def execute(state):
                """Complete the subject agent without contacting an external API."""
                observed["calls"] = observed.get("calls", 0) + 1
                await asyncio.sleep(0.1)
                return state

            return execute

        return stub()

    monkeypatch.setattr(HARNESS.inspect_swe, harness, factory)
    task, env = local_task
    env.write_file = AsyncMock()
    monkeypatch.setattr(importlib.import_module("inferencebench.harness_default"), "sandbox", lambda: env)
    task.dataset[0].metadata["agent_seconds"] = seconds
    task.solver = original_agent(
        harness=harness, version=version, continue_until_deadline=continue_until_deadline
    )
    [log] = inspect_eval(
        task,
        model="mockllm/subject",
        model_roles={"integrity": judge_model()},
        display="none",
        log_dir="logs",
    )
    assert log.status == "success"
    assert observed["version"] == version
    assert observed["cwd"] == "/home/agent/task"
    assert observed["user"] == "root"
    if harness == "claude_code":
        assert observed["permission_mode"] == "bypassPermissions"
        assert all(not call.args[1].startswith("Read /tmp/inferencebench-input-")
                   for call in env.write_file.await_args_list)
    else:
        assert "permission_mode" not in observed
    assert observed["env"] == {"BASH_MAX_TIMEOUT_MS": "36000000"}
    assert log.samples[0].metadata["prepared"]
    assert log.samples[0].scores["inference_speedup"].value == {"speedup": 2.0}
    if continue_until_deadline:
        assert observed["calls"] > 1
    else:
        assert observed["calls"] == 1


@pytest.mark.parametrize("config_name", ["default", "original"])
@pytest.mark.parametrize("harness", ["claude_code", "codex_cli", "opencode", "gemini_cli"])
def test_cli_harness_overrides(monkeypatch, config_name, harness):
    """Parse the documented CLI switches and construct real SWE agents without allocating a sandbox."""
    from click.testing import CliRunner
    from inspect_ai._cli.main import inspect as inspect_command
    from inspect_ai._eval.loader import solver_from_spec

    cli = importlib.import_module("inspect_ai._cli.eval")
    observed = {}
    if config_name == "default":
        name = "inferencebench/default_agent"
        args = {"harness": harness, "harness_args": {"version": "auto"}}
    else:
        name = "inferencebench/original_agent"
        args = {"harness": harness, "version": "auto"}

    def capture(**params):
        """Validate the resolved solver at the CLI launch boundary without executing it."""
        observed.update(params)
        assert callable(solver_from_spec(params["solver"]))
        return []

    monkeypatch.setattr(cli, "eval", capture)
    monkeypatch.setattr(cli, "model_config_to_model", Mock(return_value=judge_model()))
    result = CliRunner().invoke(
        inspect_command,
        [
            "eval", "--run-config", f"src/inferencebench/run_configs/{config_name}.yaml",
            "--model", "mockllm/subject", "--solver", name,
            *[part for key, value in args.items() for part in ("-S", f"{key}={json.dumps(value)}")],
        ],
    )
    assert result.exit_code == 0, result.output or repr(result.exception)
    assert observed["solver"].solver == name
    assert observed["solver"].args == args
    assert observed["task_args"] == load_config(f"run_configs/{config_name}.yaml")["task"]["args"]


def test_unavailable_judge_roundtrip(local_task, tmp_path):
    """Serialize missing integrity judgments as unavailable speedups instead of baseline failures."""
    task, _ = local_task
    [log] = inspect_eval(
        task,
        solver=generate(),
        model="mockllm/subject",
        model_roles={"integrity": judge_model("unparseable")},
        epochs=2,
        display="none",
        log_dir="logs",
    )
    assert log.status == "success"
    loaded = read_eval_log(log.location)
    assert math.isnan(loaded.samples[0].scores["inference_speedup"].value["speedup"])
    assert math.isnan(loaded.results.scores[0].metrics["aggregate_speedup"].value)


def test_infrastructure_error_is_not_a_score(local_task, monkeypatch, tmp_path):
    """Surface broken grading infrastructure as an errored sample and still terminate its sandbox."""
    task, env = local_task
    env.exec.side_effect = RuntimeError("transport failed")
    [log] = inspect_eval(
        task,
        solver=generate(),
        model="mockllm/subject",
        display="none",
        log_dir="logs",
    )
    assert log.status == "error"
    assert log.samples[0].error is not None
    assert not log.samples[0].scores
    env.terminate.assert_awaited_once()


def test_metrics_and_missing_epochs():
    """Use geometric scenario aggregation after arithmetic seed and epoch means without hiding missing attempts."""
    for scenario in "ABCD":
        assert performance(measurements(2), scenario) / performance(
            measurements(), scenario
        ) == pytest.approx(2)
    values = [("A", 1), ("A", 3), ("B", 8)]
    scores = [
        SampleScore(
            score=Score(value={"speedup": value}),
            sample_metadata={"scenario": scenario},
        )
        for scenario, value in values
    ]
    assert aggregate_speedup()(scores) == pytest.approx(4)
    reducer = complete_mean()
    assert (
        reducer([Score(value={"speedup": 1}), Score(value={"speedup": 3})]).value[
            "speedup"
        ]
        == 2
    )
    assert math.isnan(
        reducer(
            [Score(value={"speedup": 1}), Score(value={"speedup": math.nan})]
        ).value["speedup"]
    )


@pytest.mark.parametrize(
    "text,expected",
    [
        ("no contamination detected\nonly allowed use detected", True),
        ("contamination detected\nonly allowed use detected", False),
        (
            "no contamination detected\ncontamination detected\nonly allowed use detected",
            None,
        ),
        ("looks good", None),
    ],
)
def test_judgment_contract(text, expected):
    """Reject malformed and contradictory integrity verdicts without confusing them with prohibited behavior."""
    assert parse_judgment(text) is expected


def test_native_run_configs():
    """Accept concise configs and leave omitted optional native settings unset."""
    from inspect_ai._cli.eval import RunConfigInput
    from inspect_ai.model import GenerateConfig

    for name in ["default", "original"]:
        parsed = RunConfigInput.model_validate(
            load_config("run_configs/" + name + ".yaml")
        )
        assert parsed.task.task == "inferencebench/inference_bench"
        task = inference_bench(
            **load_config(f"run_configs/{name}.yaml")["task"]["args"]
        )
        assert task.config == GenerateConfig(**parsed.generate_config.model_dump())
        assert task.time_limit is None
        assert task.turn_limit is None
        assert task.message_limit is None
        assert task.cost_limit is None
        assert parsed.eval_config.max_samples == 1


@pytest.mark.parametrize(
    "config_defaults,seconds", [("default", 60), ("original", None)]
)
def test_independent_native_budget_defaults(
    local_task, tmp_path, config_defaults, seconds
):
    """Keep each profile's token cap when the independent optimization deadline changes."""
    args = load_config(f"run_configs/{config_defaults}.yaml")["task"]["args"]
    task = inference_bench(
        **dict(args, agent_seconds=seconds, scenarios="A", seed_pairs=[[21, 1337]])
    )
    task.sandbox = None
    assert task.token_limit == (100000000 if config_defaults == "default" else None)
    [log] = inspect_eval(
        task,
        solver=generate(),
        model="mockllm/subject",
        model_roles={"integrity": judge_model()},
        token_limit=500,
        display="none",
        log_dir="logs",
    )
    assert log.status == "success"
    assert log.eval.config.token_limit == 500
    assert log.samples[0].metadata["agent_seconds"] == seconds


def test_registered_yaml_solver(local_task, monkeypatch, tmp_path):
    """Resolve a YAML-selected registry namespace that is not an importable Python module."""

    @solver(name="review_namespace/probe")
    def probe() -> TASK.Solver:
        """Create a distinct solver whose execution is visible in the final sample."""

        async def solve(state, generate):
            """Mark the alternate solver and make a normal model call."""
            state.metadata["alternate_solver"] = True
            return await generate(state)

        return solve

    def config(path="run_configs/default.yaml"):
        """Substitute only the configured solver while preserving task defaults and metadata."""
        result = load_config(path)
        if path.endswith("default.yaml"):
            result["solver"] = {"solver": "review_namespace/probe", "args": {}}
        return result

    monkeypatch.setattr(TASK, "load_config", config)
    task = inference_bench(scenarios="A", seed_pairs=[[21, 1337]])
    task.sandbox = None
    [log] = inspect_eval(
        task,
        model="mockllm/subject",
        model_roles={"integrity": judge_model()},
        display="none",
        log_dir="logs",
    )
    assert log.status == "success"
    assert log.samples[0].metadata["alternate_solver"]
    assert log.samples[0].metadata["prepared"]


@pytest.mark.parametrize(
    "options",
    [
        {"quality_samples": 1.5},
        {"quality_samples": True},
        {"quality_seed": -1},
        {"quality_seed": True},
        {"request_limit": 1.5},
        {"max_model_len": 0},
        {"context_length": 0},
        {"context_length": -1},
        {"context_length": True},
        {"context_length": 1.5},
        {"quality_concurrency": False},
        {"agent_seconds": math.inf},
        {"server_wait_seconds": math.nan},
        {"request_timeout_seconds": -1},
        {"quality_tau": math.nan},
        {"quality_tau": True},
        {"quality_reference_backend": "sglang"},
        {"strict_prompt": "yes"},
        {"baseline_dtype": "int8"},
    ],
)
def test_invalid_workload_arguments(options):
    """Reject settings that upstream would silently replace or fail on after allocating a GPU."""
    with pytest.raises(ValueError):
        inference_bench(**options)


def test_runtime_uses_configured_workload():
    """Pass explicit maximum length, concurrency, and quality sampling values into the upstream environment."""
    from inferencebench.assets.scripts.runtime import environment

    task = inference_bench(
        max_model_len=16384, quality_concurrency=2, quality_samples=16, quality_seed=0
    )
    import os
    from unittest.mock import patch

    with patch.dict(os.environ, {"HF_HOME": "/tmp/cache"}):
        env = environment(task.dataset[0].metadata)
    assert env["INFERENCE_BENCH_MAX_MODEL_LEN"] == "16384"
    assert env["INFERENCE_BENCH_QUALITY_CONCURRENCY"] == "2"
    assert env["INFERENCE_BENCH_QUALITY_MMLUPRO_N"] == "16"
    assert env["INFERENCE_BENCH_QUALITY_SEED"] == "0"


def test_prompt_selection_contract():
    """Reject using the integrity rubric as a subject prompt or task instructions as a judge rubric."""
    with pytest.raises(ValueError, match="system_prompt"):
        inference_bench(system_prompt="original_judge")
    with pytest.raises(ValueError, match="judge_prompt"):
        SCORERS.inference_speedup(
            grader_role="integrity", judge_prompt="original", max_grader_attempts=3
        )
    assert PROMPTS["original"].role == "subject"
    assert PROMPTS["original_judge"].role == "judge"
    assert PROMPTS["original_judge"].output_format


async def test_modal_filesystem_adapter():
    """Use the supported Modal filesystem methods for writes, reads, and parent creation."""
    from inferencebench.environment import InferenceSandbox

    filesystem = SimpleNamespace(
        make_directory=SimpleNamespace(aio=AsyncMock()),
        write_bytes=SimpleNamespace(aio=AsyncMock()),
        read_bytes=SimpleNamespace(aio=AsyncMock(return_value=b"hello")),
    )
    env = InferenceSandbox(SimpleNamespace(filesystem=filesystem))
    await env._create_parent_folder("/tmp/test")
    await env._write_file_content("/tmp/test/file", "hello")
    assert await env._read_file_content("/tmp/test/file") == b"hello"
    filesystem.write_bytes.aio.assert_awaited_once_with(b"hello", "/tmp/test/file")


def test_dead_baseline_stops_readiness_polling():
    """Fail immediately when the baseline process has exited instead of keeping an idle H100 for fifteen minutes."""
    from inferencebench.assets.scripts.runtime import wait_ready

    process = SimpleNamespace(poll=lambda: 1)
    started = time.monotonic()
    assert wait_ready(process, 900) is False
    assert time.monotonic() - started < 1


@pytest.mark.parametrize(
    "failure", ["launcher", "quality", "empty", "undefined_metric", "integrity"]
)
def test_invalid_submissions_receive_baseline(local_task, tmp_path, failure):
    """Assign the original 1x result to invalid submissions while preserving their distinct failure evidence."""
    task, env = local_task
    metrics = measurements(2)
    judgment = "no contamination detected\nonly allowed use detected"
    if failure == "launcher":
        metrics = {"invalid_submission": "launcher exited"}
    elif failure == "quality":
        metrics["quality_check"]["pass"] = False
    elif failure == "empty":
        metrics["profiles"]["burst"]["success_count"] = 0
    elif failure == "undefined_metric":
        metrics["profiles"]["burst"]["ttft"]["p50"] = None
    else:
        judgment = "contamination detected\nonly allowed use detected"

    async def read_file(path):
        """Expose the selected submission failure through the actual scorer transport boundary."""
        if path.endswith("final.json"):
            return json.dumps(metrics)
        return verdict_file(path) if path.endswith("_judgement.txt") else "server log"

    env.read_file = read_file
    [log] = inspect_eval(
        task,
        solver=generate(),
        model="mockllm/subject",
        model_roles={"integrity": judge_model(judgment)},
        display="none",
        log_dir="logs",
    )
    assert log.status == "success"
    assert log.samples[0].scores["inference_speedup"].value["speedup"] == 1


@pytest.mark.parametrize(
    "submit,nudge,expected_limit",
    [(False, True, True), (False, False, False), (True, True, False)],
)
def test_token_budget_stopping(local_task, tmp_path, submit, nudge, expected_limit):
    """Stop only at the token budget by default while honoring explicit nudge and submission toggles."""
    task, _ = local_task
    task.dataset[0].metadata["agent_seconds"] = None
    task.solver = react_agent(submit=submit, nudge_prompt=nudge)
    outputs = [
        ModelOutput.for_tool_call("mockllm/subject", "submit", {"answer": "Ready"})
        if submit
        else ModelOutput.from_content("mockllm/subject", "Ready")
        for _ in range(5)
    ]
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
    assert log.status == "success"
    sample = read_eval_log(log.location).samples[0]
    assert bool(sample.limit and sample.limit.type == "token") == expected_limit
    assert "agent_deadline_reached" not in sample.metadata
    reminders = [
        m.text
        for m in sample.messages
        if m.role == "user" and "Token budget:" in m.text
    ]
    assert "0 / 500" in reminders[0]
    if expected_limit:
        assert sample.token_limit_usage >= 500
        assert any("200 / 500" in text for text in reminders)
        assert any(
            "Continue working on the inference server" in m.text
            for m in sample.messages
        )
    else:
        assert sample.token_limit_usage == 200


def test_token_reminder_after_tools(local_task, tmp_path):
    """Report updated usage after tool turns and preserve the switch that disables reminders."""
    task, _ = local_task
    task.dataset[0].metadata["agent_seconds"] = None
    outputs = [
        ModelOutput.for_tool_call("mockllm/subject", "bash", {}),
        ModelOutput.from_content("mockllm/subject", "Ready"),
    ]
    for output in outputs:
        output.usage = ModelUsage(input_tokens=100, output_tokens=100, total_tokens=200)

    task.solver = react_agent(nudge_prompt=False)
    [log] = inspect_eval(
        task,
        model=get_model("mockllm/subject", custom_outputs=outputs),
        model_roles={"integrity": judge_model()},
        token_limit=500,
        display="none",
        log_dir="logs",
    )
    assert log.status == "success"
    assert any(
        "200 / 500" in m.text for m in log.samples[0].messages if m.role == "user"
    )
    assert not log.samples[0].limit


@pytest.mark.parametrize(
    "contamination,allowed,expected",
    [
        ("no contamination detected", "only allowed use detected", True),
        ("contamination detected", "only allowed use detected", False),
        ("no contamination detected", "disallowed use detected", False),
    ],
)
def test_filename_labeled_judgments(contamination, allowed, expected):
    """Accept the original filename-labeled verdict format observed in the live integrity judge."""
    text = f"**`contamination_judgement.txt`:** {contamination}\n\n**`disallowed_model_judgement.txt`:** {allowed}"
    assert parse_judgment(text) is expected
    assert (
        parse_judgment(text + "\ncontamination detected\nno contamination detected")
        is None
    )


def test_errored_epoch_coverage(local_task, monkeypatch, tmp_path):
    """Expose omitted infrastructure failures beside the finite aggregate in an error-tolerant run."""
    task, env = local_task

    async def restart(state, include_transcript):
        """Fail only the second epoch while preserving the first epoch's valid measurement."""
        if state.epoch == 2:
            raise RuntimeError("second epoch transport failed")
        return env

    monkeypatch.setattr(SCORERS, "restart_for_scoring", restart)
    [log] = inspect_eval(
        task,
        solver=generate(),
        model="mockllm/subject",
        model_roles={"integrity": judge_model()},
        epochs=2,
        fail_on_error=False,
        display="none",
        log_dir="logs",
    )
    loaded = read_eval_log(log.location)
    assert loaded.status == "success"
    assert loaded.results.total_samples == 2
    assert loaded.results.completed_samples == 1
    assert sum(sample.error is not None for sample in loaded.samples) == 1
    metrics = {
        name: metric
        for result in loaded.results.scores
        for name, metric in result.metrics.items()
    }
    assert metrics["aggregate_speedup"].value == 2
    assert metrics["scored_attempts"].value == 1
    assert metrics["unscored_attempts"].value == 0


def test_bound_judge_configuration(local_task, tmp_path):
    """Preserve the explicit judge model's generation settings and record its identity and attempt count."""
    from inspect_ai.model import GenerateConfig

    task, _ = local_task
    judge = get_model(
        "mockllm/alternate",
        config=GenerateConfig(temperature=0.37, extra_body={"test_marker": "judge"}),
        custom_outputs=[
            ModelOutput.from_content(
                "mockllm/alternate",
                "no contamination detected\nonly allowed use detected",
            )
        ],
    )
    [log] = inspect_eval(
        task,
        solver=generate(),
        model="mockllm/subject",
        model_roles={"integrity": judge},
        temperature=0.82,
        display="none",
        log_dir="logs",
    )
    assert log.status == "success"
    events = [
        event
        for event in read_eval_log(log.location, resolve_attachments=True).samples[0].events
        if event.event == "model" and event.model == "mockllm/alternate"
    ]
    assert events and all(event.config.temperature == 0.37 for event in events)
    assert "InferenceBench Scenario A: Input heavy (prefill speed)" in events[0].input[0].text
    assert all(event.config.extra_body == {"test_marker": "judge"} for event in events)
    assert log.samples[0].scores["inference_speedup"].metadata["integrity_judge"] == {
        "model": "mockllm/alternate",
        "role": "integrity",
        "attempts": 1,
        "include_transcript": True,
        "transcript_hint": True,
        "preloaded_evidence": ["start_server.sh", "server.log"],
        "judge_cli_version": "2.1.114",
        "verdict_files": {
            "contamination_judgement.txt": "no contamination detected\n",
            "disallowed_model_judgement.txt": "only allowed use detected\n",
        },
    }
    [call] = JUDGE_CALLS
    assert str(call["model"]) == "mockllm/alternate" and call["version"] == "2.1.114"


def test_plain_and_default_configuration(local_task, tmp_path):
    """Observe equivalent task inputs and native settings for a plain task and its explicit default arguments."""
    tasks = [inference_bench(), inference_bench(**load_config()["task"]["args"])]
    logs = []
    for task in tasks:
        task.sandbox = None
        [log] = inspect_eval(
            task,
            solver=generate(),
            model="mockllm/subject",
            model_roles={"integrity": judge_model()},
            limit=1,
            display="none",
            log_dir="logs",
        )
        assert log.status == "success"
        logs.append(log)
    assert logs[0].samples[0].input == logs[1].samples[0].input
    assert logs[0].eval.config == logs[1].eval.config
    assert (
        logs[0].samples[0].scores["inference_speedup"].value
        == logs[1].samples[0].scores["inference_speedup"].value
    )


def test_alternate_scorer_and_reducer(local_task, monkeypatch, tmp_path):
    """Honor a configured scorer factory and an explicit native epoch reducer without replacing setup."""
    from inspect_ai import Epochs
    from inspect_ai.scorer import Scorer, max_score, scorer

    from inferencebench.metrics import aggregate_speedup

    @scorer(metrics=[aggregate_speedup()])
    def alternate() -> Scorer:
        """Create a compatible speedup scorer with different values in independent epochs."""

        async def score(state, target):
            """Return a controlled speedup only after the shared environment has been prepared."""
            assert state.metadata["prepared"]
            return Score(value={"speedup": float(state.epoch)})

        return score

    monkeypatch.setattr(SCORERS, "alternate", alternate, raising=False)
    task = inference_bench(
        scenarios="A",
        seed_pairs=[[21, 1337]],
        scorer={"name": "inferencebench.scorers.alternate", "args": {}},
    )
    task.sandbox = None
    [log] = inspect_eval(
        task,
        solver=generate(),
        model="mockllm/subject",
        epochs=Epochs(2, max_score()),
        display="none",
        log_dir="logs",
    )
    assert log.status == "success"
    assert log.results.scores[0].metrics["aggregate_speedup"].value == 2


def test_registered_agent_config(local_task, tmp_path):
    """Adapt a registered agent selected in the solver config while preserving the shared scorer."""
    from inspect_ai.agent import Agent, agent
    from inspect_ai.model import ChatMessageAssistant

    @agent(name="review_namespace/agent_probe")
    def probe() -> Agent:
        """Create a registered agent in a namespace that is not a Python module."""

        async def execute(state):
            """Leave an observable response through the native agent-to-solver bridge."""
            state.messages.append(
                ChatMessageAssistant(content="Configured agent executed")
            )
            return state

        return execute

    task, _ = local_task
    task.solver = TASK._solver_from_config(
        {"solver": "review_namespace/agent_probe", "args": {}}
    )
    [log] = inspect_eval(
        task,
        model="mockllm/subject",
        model_roles={"integrity": judge_model()},
        display="none",
        log_dir="logs",
    )
    assert log.status == "success"
    assert any(
        message.text == "Configured agent executed"
        for message in log.samples[0].messages
    )
    assert log.samples[0].scores["inference_speedup"].value == {"speedup": 2.0}


def test_native_failure_and_limit_defaults(monkeypatch):
    """Forward all supported task-layer limits and failure settings from the selected YAML profile."""
    expected = {
        "message_limit": 10,
        "cost_limit": 2.5,
        "fail_on_error": False,
        "continue_on_fail": True,
        "score_on_error": False,
    }

    def config(path="run_configs/default.yaml"):
        """Override native YAML values without changing the public task arguments."""
        result = load_config(path)
        if path.endswith("default.yaml"):
            result["eval_config"].update(expected)
        return result

    monkeypatch.setattr(TASK, "load_config", config)
    task = inference_bench()
    for name, value in expected.items():
        assert getattr(task, name) == value


def test_registered_solver_error_is_preserved():
    """Preserve a valid solver factory's own error instead of treating it as a missing agent."""

    @solver(name="review_namespace/broken_factory")
    def broken() -> TASK.Solver:
        """Reproduce a configuration error inside an existing solver factory."""
        raise KeyError("missing-setting")

    with pytest.raises(KeyError, match="missing-setting"):
        TASK._solver_from_config(
            {"solver": "review_namespace/broken_factory", "args": {}}
        )


@pytest.mark.parametrize("profile", ["default", "original"])
def test_solver_error_still_scores_submission(local_task, profile):
    """Grade a partial submission and retain the solver failure in the durable log."""
    _, env = local_task
    task = inference_bench(config_defaults=profile, scenarios="A", seed_pairs=[[21, 1337]])
    task.sandbox = None

    @solver
    def failing_agent():
        async def solve(state, generate):
            await generate(state)
            raise RuntimeError("agent failed after writing a usable submission")
        return solve

    [log] = inspect_eval(
        task, solver=failing_agent(), model="mockllm/subject",
        model_roles={"integrity": judge_model()}, display="none", log_dir="logs",
    )
    sample = read_eval_log(log.location).samples[0]
    assert sample.error is not None
    assert "agent failed after writing a usable submission" in sample.error.message
    assert sample.scores["inference_speedup"].value == {"speedup": 2.0}
    env.terminate.assert_awaited_once()


def test_snapshot_error_preserves_submission(local_task, monkeypatch, tmp_path):
    """Save the submission before a failed scoring restart, without fabricating a grade."""
    environment = importlib.import_module("inferencebench.environment")
    from inspect_ai.hooks._hooks import get_all_hooks

    monkeypatch.setenv("HAWK_JOB_ID", "local-restart-error-test")
    hook = next(h for h in get_all_hooks() if isinstance(h, environment.HawkArtifacts))
    destination = f"memory://inferencebench-test/{tmp_path.name}"
    monkeypatch.setattr(hook, "destinations", {None: destination})
    task, env = local_task
    submission = b"saved submission archive"

    async def download(remote, local):
        Path(local).write_bytes(submission)

    env.download = AsyncMock(side_effect=download)

    async def restart(config):
        assert (tmp_path / "submission.tar.gz").read_bytes() == submission
        assert (tmp_path / "agent-transcript.json").exists()
        raise TimeoutError("RunPod command exceeded its timeout")

    env.restart = AsyncMock(side_effect=restart)
    monkeypatch.setattr(environment, "gpu_environment", lambda: env)
    monkeypatch.setattr(SCORERS, "restart_for_scoring", environment.restart_for_scoring)
    [log] = inspect_eval(
        task, solver=generate(), model="mockllm/subject", display="none", log_dir="logs",
    )
    sample = read_eval_log(log.location).samples[0]
    assert log.status == "error"
    assert "RunPod command exceeded its timeout" in sample.error.message
    assert not sample.scores
    env.download.assert_awaited_once()
    fs, path = environment.url_to_fs(f"{destination}/artifacts/{sample.uuid}")
    assert fs.cat(f"{path}/submission.tar.gz") == submission
    assert fs.exists(f"{path}/agent-transcript.json")
    fs.rm(path, recursive=True)


def test_replacement_cleanup_after_artifact_failure(local_task, monkeypatch):
    """Release the replacement sandbox if host artifact bookkeeping fails before evaluator restoration."""
    environment = importlib.import_module("inferencebench.environment")
    task, env = local_task
    env.resource_id = "replacement-sandbox"
    original = SimpleNamespace(restart=AsyncMock(return_value=env))
    monkeypatch.setattr(environment, "gpu_environment", lambda: original)
    monkeypatch.setattr(SCORERS, "restart_for_scoring", environment.restart_for_scoring)
    write_text = Path.write_text

    def write(path, *args, **kwargs):
        """Fail only the post-restart artifact write after baseline setup succeeds."""
        if path.name == "scoring-sandbox.json":
            raise OSError("disk full")
        return write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", write)
    [log] = inspect_eval(
        task, solver=generate(), model="mockllm/subject", display="none", log_dir="logs"
    )
    assert log.status == "error"
    env.terminate.assert_awaited_once()
    assert not log.samples[0].scores


@pytest.mark.parametrize(
    "config_defaults,override",
    [("default", None), ("original", None), ("default", False), ("original", False)],
)
@pytest.mark.parametrize("compacted", [False, True])
@pytest.mark.parametrize("hawk", [False, True])
def test_judge_transcript_toggle(
    local_task, monkeypatch, tmp_path, config_defaults, override, compacted, hawk
):
    """Exercise both configs and overrides through transcript export, actual file reads, and a mock judge."""
    environment = importlib.import_module("inferencebench.environment")
    _, env = local_task
    if hawk:
        from inspect_ai.hooks._hooks import get_all_hooks
        monkeypatch.setenv("HAWK_JOB_ID", "local-artifact-test")
        hook = next(h for h in get_all_hooks() if isinstance(h, environment.HawkArtifacts))
        destination = f"memory://inferencebench-test/{tmp_path.name}"
        monkeypatch.setattr(hook, "destinations", {None: destination})
    args = load_config(f"run_configs/{config_defaults}.yaml")["task"]["args"]
    if override is not None:
        args["scorer"]["args"]["include_transcript"] = override
        args["scorer"]["args"]["transcript_hint"] = override
    enabled = args["scorer"]["args"]["include_transcript"]
    hinted = args["scorer"]["args"]["transcript_hint"]
    task = inference_bench(**{**args, "scenarios": "A", "seed_pairs": [[21, 1337]]})
    task.sandbox = None

    def local_path(remote):
        """Map sandbox evidence paths into this test's isolated filesystem."""
        return tmp_path / "sandbox" / remote.lstrip("/")

    async def write_file(path, content):
        """Persist the real scorer's exported evidence for inspection by the mock judge."""
        path = local_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content if isinstance(content, bytes) else content.encode())

    async def upload(local, remote):
        """Transfer the streamed evidence while ignoring the fixture's synthetic evaluator archive."""
        if remote.endswith("agent-transcript.json"):
            await write_file(remote, Path(local).read_text())

    async def execute(command, **kwargs):
        """Run evidence reads and stale-file removal locally while substituting GPU-only infrastructure."""
        archive = command[0] == "tar" and "-czf" in command
        if command[0] not in {"rm", "sed", "ls"} and not archive:
            return ExecResult(success=True, returncode=0, stdout="", stderr="")
        command = [
            str(local_path(arg)) if arg.startswith("/") else arg for arg in command
            if arg != "--ignore-failed-read"
        ]
        # BSD sed lacks GNU's separator; mapped fixture paths are always absolute.
        if command[0] == "sed":
            command.remove("--")
        process = await asyncio.create_subprocess_exec(
            *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        return ExecResult(
            process.returncode == 0,
            process.returncode,
            stdout.decode(),
            stderr.decode(),
        )

    transcript = f"{environment.REMOTE}/agent-transcript.json"
    local_path(transcript).parent.mkdir(parents=True)
    local_path(transcript).write_text("stale transcript from a previous snapshot")
    launcher = "/home/agent/task/start_server.sh"
    local_path(launcher).parent.mkdir(parents=True)
    local_path(launcher).write_text("launcher evidence")
    env.resource_id = "local-scoring-sandbox"
    env.restart = AsyncMock(return_value=env)
    env.upload = AsyncMock(side_effect=upload)
    fixture_read = env.read_file

    async def read_file(path):
        """Serve evidence from the mapped filesystem when present, otherwise the fixture's outputs."""
        local = local_path(path)
        return local.read_text() if local.is_file() else await fixture_read(path)

    env.read_file = read_file
    async def download(remote, local):
        Path(local).write_bytes(local_path(remote).read_bytes())
    env.download = AsyncMock(side_effect=download)
    env.write_file = AsyncMock(side_effect=write_file)
    env.exec.side_effect = execute
    monkeypatch.setattr(environment, "gpu_environment", lambda: env)
    monkeypatch.setattr(SCORERS, "restart_for_scoring", environment.restart_for_scoring)
    judge = get_model(
        "mockllm/transcript-judge",
        custom_outputs=[
            ModelOutput.from_content(
                "mockllm/transcript-judge",
                "no contamination detected\nonly allowed use detected",
            )
        ],
    )
    @solver
    def subject():
        """Simulate context compaction after an observable model action."""
        async def solve(state, generate):
            """Keep the event record while replacing the active context with a summary."""
            from inspect_ai.model import ChatMessageAssistant, ChatMessageTool
            from inspect_ai.tool import ToolCall

            # CLI tools arrive through the next bridged model input, without ToolEvents.
            state.messages.extend([
                ChatMessageAssistant(content="", tool_calls=[ToolCall(
                    id="cli-tool", function="Bash", arguments={"command": "printf evidence"},
                )]),
                ChatMessageTool(content="CLI tool result evidence " + "x" * 3000, tool_call_id="cli-tool"),
            ])
            state = await generate(state)
            await get_model().generate(state.messages)
            if compacted:
                from inspect_ai.model import ChatMessageUser
                state.messages = [ChatMessageUser(content="Compacted context without the earlier evidence")]
            return state
        return solve

    [log] = inspect_eval(
        task,
        solver=subject(),
        model=get_model(
            "mockllm/subject",
            custom_outputs=[
                ModelOutput.from_content(
                    "mockllm/subject", "subject transcript evidence"
                )
            ] * 2,
        ),
        model_roles={"integrity": judge},
        display="none",
        log_dir="logs",
    )
    assert log.status == "success", log.error
    sample = read_eval_log(log.location, resolve_attachments=True).samples[0]
    initial = next(
        event
        for event in sample.events
        if event.event == "model" and event.model == "mockllm/transcript-judge"
    )
    assert ("agent-transcript.json" in initial.input[0].text) is hinted
    assert "## Contents of `start_server.sh`\n```\nlauncher evidence\n```" in initial.input[0].text
    assert local_path(transcript).exists() is enabled
    if enabled:
        evidence = json.loads(local_path(transcript).read_text())
        assert evidence["messages"] == [
            message.model_dump(mode="json") for message in sample.messages
        ]
        assert "subject transcript evidence" in json.dumps(evidence["events"])
        assert json.dumps(evidence["events"]).count("CLI tool result evidence") == 1
    else:
        assert not any(
            call.args[0] == transcript for call in env.write_file.await_args_list
        )
    assert (
        sample.scores["inference_speedup"].metadata["integrity_judge"][
            "include_transcript"
        ]
        is enabled
    )
    env.terminate.assert_awaited_once()
    if hawk:
        import io
        import tarfile
        fs, path = environment.url_to_fs(f"{destination}/artifacts/{sample.uuid}")
        assert fs.exists(f"{path}/trusted/speed/baseline_metrics.json")
        assert fs.exists(f"{path}/final.json")
        with tarfile.open(fileobj=io.BytesIO(fs.cat(f"{path}/submission.tar.gz"))) as archive:
            assert archive.extractfile("task/start_server.sh").read() == b"launcher evidence"
        assert not local_path(f"{environment.REMOTE}/submission.tar.gz").exists()
        assert fs.exists(f"{path}/agent-transcript.json") is enabled
        fs.rm(path, recursive=True)


@pytest.mark.parametrize("value", [None, "false", 0])
@pytest.mark.parametrize("setting", ["include_transcript", "transcript_hint"])
def test_invalid_judge_toggles(setting, value):
    """Reject ambiguous judge evidence settings before creating a scoring sandbox."""
    args = load_config()["task"]["args"]["scorer"]["args"]
    with pytest.raises(ValueError, match=f"{setting} must be a boolean"):
        SCORERS.inference_speedup(**{**args, setting: value})


@pytest.mark.parametrize("version", ["auto", "latest", "", None])
def test_judge_requires_exact_cli_version(version):
    """Refuse mutable Claude Code tags for the judge so verdicts stay reproducible."""
    args = load_config()["task"]["args"]["scorer"]["args"]
    with pytest.raises(ValueError, match="exact Claude Code version"):
        SCORERS.inference_speedup(**{**args, "judge_cli_version": version})


def test_transcript_hint_requires_export():
    """Refuse to point the judge at a transcript that will not exist in the scoring sandbox."""
    args = load_config()["task"]["args"]["scorer"]["args"]
    with pytest.raises(ValueError, match="transcript_hint requires include_transcript"):
        SCORERS.inference_speedup(**{**args, "include_transcript": False, "transcript_hint": True})


@pytest.mark.parametrize("missing_launcher", [False, True])
def test_original_judge_runs_claude_code_with_upstream_prompt(local_task, monkeypatch, missing_launcher):
    """Give the original-config judge upstream's rendered prompt inside Claude Code, started in the restarted submission like upstream."""
    from inspect_ai.model import ChatMessageUser

    _, env = local_task
    launcher = "/home/agent/task/start_server.sh"
    log_lines = [f"server line {index}" for index in range(1, 251)]

    async def read_file(path):
        """Serve the restarted submission's launcher, a 250-line final server log, and the judge's verdict files."""
        if path.endswith("final.json"):
            return json.dumps(measurements(2))
        if path == launcher:
            if missing_launcher:
                raise FileNotFoundError(path)
            return "launcher evidence"
        if path.endswith("final-server.log"):
            return "\n".join(log_lines)
        return verdict_file(path)

    env.read_file = read_file
    args = load_config("run_configs/original.yaml")["task"]["args"]
    assert args["scorer"]["args"] == {**args["scorer"]["args"], "max_grader_attempts": 1, "include_transcript": True, "transcript_hint": False, "judge_cli_version": "2.1.114"}
    task = inference_bench(**{**args, "scenarios": "A", "seed_pairs": [[21, 1337]], "agent_seconds": 2})
    task.sandbox = None
    judge = get_model(
        "mockllm/original-judge",
        custom_outputs=[ModelOutput.from_content("mockllm/original-judge", "no contamination detected\nonly allowed use detected")],
    )
    [log] = inspect_eval(task, solver=generate(), model="mockllm/subject", model_roles={"integrity": judge}, display="none", log_dir="logs")
    assert log.status == "success", log.error
    sample = read_eval_log(log.location, resolve_attachments=True).samples[0]
    initial = next(e for e in sample.events if e.event == "model" and e.model == "mockllm/original-judge")
    text = initial.input[0].text
    assert isinstance(initial.input[0], ChatMessageUser)
    rubric = PROMPTS["original_judge"].prompt.replace("{model}", "mistralai/Mistral-7B-Instruct-v0.3").replace("{benchmark}", "InferenceBench Scenario A: Input heavy (prefill speed)")
    assert text.startswith(rubric) and "Inspect adapter" not in text and "run_shell" not in text and "agent-transcript.json" not in text
    assert "# Pre-loaded File Evidence\nThe following files were found in the task directory. Use them as primary evidence for your judgement." in text
    assert ("## Contents of `start_server.sh`\n```\nlauncher evidence\n```" in text) is not missing_launcher
    assert "## Contents of `server.log`\n```\n[...truncated 50 lines...]\nserver line 51\n" in text
    assert "server line 50\n" not in text and text.rstrip().count("server line 250") == 1
    [call] = JUDGE_CALLS
    assert str(call["model"]) == "mockllm/original-judge" and call["version"] == "2.1.114" and call["cwd"] == "/home/agent/task"
    assert call["user"] == "root" and call["sandbox"] == "default" and call["permission_mode"] == "bypassPermissions" and call["retry_refusals"] == 0
    verdicts = [f"/home/agent/task/{name}" for name in SCORERS.VERDICT_FILES]
    assert any(list(c.args[0]) == ["rm", "-f", *verdicts] for c in env.exec.await_args_list)
    details = sample.scores["inference_speedup"].metadata["integrity_judge"]
    assert details["preloaded_evidence"] == (["server.log"] if missing_launcher else ["start_server.sh", "server.log"])
    assert details["attempts"] == 1 and details["transcript_hint"] is False and sorted(details["verdict_files"]) == SCORERS.VERDICT_FILES
    assert sample.scores["inference_speedup"].value == {"speedup": 2.0}


@pytest.mark.parametrize("failure", ["solver", "connection"])
def test_failed_solver_retains_submission(local_task, monkeypatch, tmp_path, failure):
    """Keep an unfinished launcher and model evidence when the agent fails before scoring."""
    import io
    import tarfile

    from inspect_ai.hooks._hooks import get_all_hooks

    environment = importlib.import_module("inferencebench.environment")
    task, env = local_task
    monkeypatch.setenv("HAWK_JOB_ID", "failed-agent-test")
    destination = f"memory://failed-agent/{tmp_path.name}"
    hook = next(h for h in get_all_hooks() if isinstance(h, environment.HawkArtifacts))
    monkeypatch.setattr(hook, "destinations", {None: destination})
    monkeypatch.setattr(environment, "gpu_environment", lambda: env)
    workspace = tmp_path / "sandbox" / "task"
    workspace.mkdir(parents=True)
    launcher = workspace / "start_server.sh"
    launcher.write_text("unfinished launcher")

    async def download(remote, local):
        # The transport fixture serves a real archive of the agent's working files.
        with tarfile.open(local, "w:gz") as archive:
            archive.add(workspace, arcname="task")

    env.download = AsyncMock(side_effect=download)

    @solver
    def broken_agent():
        async def solve(state, generate):
            await generate(state)
            if failure == "connection":
                import asyncssh
                from asyncssh.misc import async_context_manager
                from inspect_ai.util._sandbox.events import SandboxEnvironmentProxy

                from inferencebench.environment import RunPodSandbox

                config = load_config("assets/sandboxes/runpod.yaml")
                config["poll_interval_seconds"] = 0
                provider = RunPodSandbox(config)
                @async_context_manager
                async def failed_connect(*args, **kwargs):
                    raise TimeoutError()

                monkeypatch.setattr(asyncssh, "connect", failed_connect)
                await SandboxEnvironmentProxy(provider).exec(["true"])
            raise RuntimeError("native agent died")
        return solve

    [log] = inspect_eval(task, solver=broken_agent(), model="mockllm/subject",
                        model_roles={"integrity": judge_model()}, display="none", log_dir="logs")
    assert log.status == "error"
    sample = read_eval_log(log.location).samples[0]
    assert ("native agent died" if failure == "solver" else "SSH connection failed") in sample.error.message
    assert sample.scores["inference_speedup"].value == {"speedup": 2.0}
    fs, path = environment.url_to_fs(f"{destination}/artifacts/{sample.uuid}")
    with tarfile.open(fileobj=io.BytesIO(fs.cat(f"{path}/submission.tar.gz"))) as archive:
        assert archive.extractfile("task/start_server.sh").read() == b"unfinished launcher"
    assert json.loads(fs.cat(f"{path}/agent-transcript.json"))["events"]
    fs.rm(path, recursive=True)


@pytest.mark.parametrize("config_defaults", ["default", "original"])
@pytest.mark.parametrize("strict", [True, False])
def test_strict_prompt_toggle(config_defaults, strict):
    """Insert the leaderboard's strict rules after the base-model constraint in both prompts, or leave the text untouched."""
    args = load_config(f"run_configs/{config_defaults}.yaml")["task"]["args"]
    assert args["strict_prompt"] is True
    text = inference_bench(**{**args, "scenarios": "A", "seed_pairs": [[21, 1337]], "strict_prompt": strict}).dataset[0].input
    anchor = "* Base Model: You must use mistralai/Mistral-7B-Instruct-v0.3."
    rules = "* Checkpoints: Do not download or serve third-party pre-quantized, distilled, or fine-tuned checkpoints of mistralai/Mistral-7B-Instruct-v0.3"
    harness = "* Evaluation harness: Do not modify, replace, wrap, or bypass evaluate.py"
    assert anchor in text and "{model}" not in text
    assert (rules in text) is strict and (harness in text) is strict
    if strict:
        assert text.index(anchor) < text.index(rules) < text.index(harness) < text.index("## 5. Output Contract")
        assert text.replace(PROMPTS["strict_rules"].prompt.replace("{model}", "mistralai/Mistral-7B-Instruct-v0.3") + "\n", "") == inference_bench(
            **{**args, "scenarios": "A", "seed_pairs": [[21, 1337]], "strict_prompt": False}
        ).dataset[0].input


async def test_deadline_absorbs_failures_after_expiry(monkeypatch):
    """Treat any failure surfacing once the budget has expired as the budget ending, and re-raise earlier failures."""
    from inferencebench import harness_default as reminders

    state = SimpleNamespace(metadata={})
    end = time.time() + 0.2
    monkeypatch.setattr(reminders, "store", lambda: SimpleNamespace(get=lambda key: end))

    async def cut_off(state, generate):
        """Mimic asyncssh turning the deadline's cancellation into an empty RuntimeError."""
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            raise RuntimeError("") from None

    assert await reminders.with_deadline(cut_off)(state, None) is state
    assert state.metadata["agent_deadline_reached"] is True

    end = time.time() + 10

    async def broken(state, generate):
        """Fail well before the deadline."""
        raise RuntimeError("real failure")

    with pytest.raises(RuntimeError, match="real failure"):
        await reminders.with_deadline(broken)(state, None)




def upstream_options(**overrides):
    """Sample options for runtime tests, taken from the packaged configuration."""
    return {**inference_bench(scenarios="A", seed_pairs=[[21, 1337]], **overrides).dataset[0].metadata, "directory": "inference_scenario_a_input_heavy"}


def test_vendored_upstream_matches_pinned_commit():
    """Keep the vendored upstream copy byte-identical to the pinned commit; every change lives in patches/."""
    from inferencebench import vendored

    lock = vendored.upstream_lock()
    assert vendored.git_tree_hash(vendored.UPSTREAM) == lock["tree"]
    assert len(lock["commit"]) == 40 and lock["source"].startswith("https://github.com/")
    assert [patch.name for patch in vendored.PATCHES] == ["0001-repair-head-truncation-boundary.patch", "0002-seed-poisson-arrivals.patch", "0003-cap-speed-output-tokens.patch"]
    assert vendored.scenario_directories() == {
        "A": "inference_scenario_a_input_heavy", "B": "inference_scenario_b_output_heavy",
        "C": "inference_scenario_c_high_load", "D": "inference_scenario_d_general",
    }


def test_original_prompt_matches_upstream_renderer(tmp_path):
    """Render the original prompt exactly as upstream's get_prompt.py does for a two-hour run."""
    import os
    import subprocess

    from inferencebench.vendored import UPSTREAM

    rendered = subprocess.run(
        [sys.executable, "src/eval/general/get_prompt.py", "--agent", "opencode", "--base-model", "mistralai/Mistral-7B-Instruct-v0.3",
         "--scenario-id", "inference_scenario_d_general", "--num-hours", "2", "--starting-point", "default"],
        cwd=UPSTREAM, capture_output=True, text=True, check=True,
        env={**os.environ, "INFERENCE_BENCH_METRICS_PATH": "/home/agent/task/metrics_preview.json"},
    ).stdout.rstrip("\n")
    args = load_config("run_configs/original.yaml")["task"]["args"]
    prompt = inference_bench(**{**args, "scenarios": "D", "seed_pairs": [[21, 1337]], "strict_prompt": False}).dataset[0].input
    assert prompt == rendered
    assert "Time Budget: 2 hours." in prompt and "metrics_preview.json" in prompt


def load_patched_runner(tmp_path, monkeypatch, patched):
    """Import upstream's sampler from a scratch package, optionally with the port's patches applied."""
    import subprocess

    from inferencebench.vendored import PATCHES, UPSTREAM

    # A distinct package name keeps these imports out of later tests' `inference` stubs.
    package = tmp_path / ("patched" if patched else "pristine") / f"inference_{'patched' if patched else 'pristine'}"
    package.mkdir(parents=True)
    for name in ["__init__.py", "runner.py", "quality_gate.py"]:
        (package / name).write_bytes((UPSTREAM / "src/eval/inference" / name).read_bytes())
    if patched:
        # The patches address src/eval/inference/runner.py; apply them in order against the copied package path.
        for patch in PATCHES:
            text = patch.read_text().replace("src/eval/inference/", f"{package.name}/")
            (tmp_path / "patched" / "runner.patch").write_text(text)
            subprocess.run(["git", "apply", "runner.patch"], cwd=tmp_path / "patched", check=True)
    for module in ["aiohttp", "numpy", "requests"]:
        monkeypatch.setitem(sys.modules, module, SimpleNamespace(ClientSession=object))
    monkeypatch.syspath_prepend(str(package.parent))
    return importlib.import_module(f"{package.name}.runner"), package


@pytest.mark.parametrize("patched", [False, True])
def test_upstream_patch_repairs_boundary_truncation(tmp_path, monkeypatch, patched):
    """Upstream's sampler aborts when decoding drops a token at the head-truncation boundary; the patch re-truncates instead."""
    runner, package = load_patched_runner(tmp_path, monkeypatch, patched)

    class Tokenizer:
        """Tokenize per character, with one character that decoding normalizes away."""

        def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
            """Count the characters of every message."""
            return list("".join(message.get("content", "") for message in messages))

        def encode(self, text, add_special_tokens=False):
            """Return one token per character."""
            return list(text)

        def decode(self, tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False):
            """Drop the normalized character, as tokenizer detokenization can at a slice boundary."""
            return "".join(tokens).replace("~", "")

    pool = package / "baselines/samples/longbench_v2/7_503"
    pool.mkdir(parents=True)
    pool.joinpath("samples.jsonl").write_text(json.dumps({"sample_id": "doc", "messages": [{"role": "user", "content": "x" * 99 + "~" + "y" * 50}]}) + "\n")
    config = {"synthetic": {"input_len": 100, "output_len": 10, "range_ratio": 1.0}, "num_requests": 1, "dataset_seed": 7}
    if not patched:
        with pytest.raises(RuntimeError, match="realized outside input range"):
            runner._prepare_requests(config, 1, Tokenizer(), None)
        return
    [request], _ = runner._prepare_requests(config, 1, Tokenizer(), None)
    assert request["input_token_count"] == 100 and "~" not in request["messages"][0]["content"]


def test_patch_seeds_poisson_arrivals(tmp_path, monkeypatch):
    """Repeat the Poisson schedule for one arrival seed, vary it across seeds, and stay unseeded without one."""
    runner, _ = load_patched_runner(tmp_path, monkeypatch, True)

    def schedule(seed):
        """Draw scenario C's Poisson arrivals under the given seed setting."""
        monkeypatch.setenv("INFERENCE_BENCH_ARRIVAL_SEED", seed)
        return runner._schedule("poisson", 256, 32)

    assert schedule("21") == schedule("21") != schedule("1337")
    assert schedule("") != schedule("")
    assert runner._schedule("constant", 4, 16) == [0, 1 / 16, 2 / 16, 3 / 16]


def test_patch_caps_speed_output_tokens(tmp_path, monkeypatch):
    """Cap forced speed outputs only when the harness sets a cap, never raising a shorter sampled length."""
    runner, _ = load_patched_runner(tmp_path, monkeypatch, True)

    monkeypatch.delenv("INFERENCE_BENCH_OUTPUT_TOKEN_CAP", raising=False)
    assert runner._output_token_cap(900) == 900
    monkeypatch.setenv("INFERENCE_BENCH_OUTPUT_TOKEN_CAP", "16")
    assert runner._output_token_cap(900) == 16 and runner._output_token_cap(8) == 8


def test_workspace_installs_upstream_files_and_environment(monkeypatch, tmp_path):
    """Install upstream's unchanged launch scaffold and evaluator stub, and export the agent environment for shells."""
    from inferencebench.assets.scripts import runtime
    from inferencebench.vendored import UPSTREAM

    task = tmp_path / "task"
    task.mkdir()
    monkeypatch.setattr(runtime, "ROOT", UPSTREAM)
    monkeypatch.setattr(runtime, "TASK", task)
    monkeypatch.setattr(runtime, "PROFILE", tmp_path / "profile.d" / "inferencebench.sh")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    options = upstream_options(base_model="org/other model", max_model_len=8192, agent_seconds=7200)
    runtime.install_workspace(options)
    assert (task / "start_server.sh").read_bytes() == (UPSTREAM / "src/eval/tasks/_shared/task_context/start_server.sh").read_bytes()
    assert (task / "test_server.sh").stat().st_mode & 0o111
    stub = (task / "evaluate.py").read_text()
    assert stub.startswith("#!/usr/bin/env python3\n") and 'sys.path.insert(0, "/opt")' in stub and "from inference_eval.runner import build_parser, run_evaluation" in stub
    assert stub in (UPSTREAM / "src/run_task.sh").read_text()
    for name in ["scenario.json", "mission.txt", "benchmark.txt"]:
        assert (task / name).read_bytes() == (UPSTREAM / "src/eval/tasks/inference_scenario_a_input_heavy" / name).read_bytes()
    profile = (tmp_path / "profile.d" / "inferencebench.sh").read_text()
    assert "export INFERENCE_BENCH_BASE_MODEL='org/other model'\n" in profile
    assert "export INFERENCE_BENCH_MAX_MODEL_LEN=8192\n" in profile
    assert "export INFERENCE_BENCH_SCENARIO=inference_scenario_a_input_heavy\n" in profile
    assert "export INFERENCE_BENCH_DATASET_SEED=21\n" in profile and "export INFERENCE_BENCH_EVAL_SEED=1337\n" in profile
    assert f"export INFERENCE_BENCH_METRICS_PATH={task}/metrics_preview.json\n" in profile
    assert "export HOST=127.0.0.1\n" in profile and "export NUM_HOURS=2\n" in profile
    assert "export INFERENCE_BENCH_ARRIVAL_SEED=21\n" in profile
    original = load_config("run_configs/original.yaml")["task"]["args"]
    assert "INFERENCE_BENCH_ARRIVAL_SEED" not in runtime.environment({**options, "seeded_arrivals": original["seeded_arrivals"]})
    # Only Scenario A's forced outputs are capped, and the original configuration keeps upstream's lengths.
    assert "export INFERENCE_BENCH_OUTPUT_TOKEN_CAP=16\n" in profile
    assert "INFERENCE_BENCH_OUTPUT_TOKEN_CAP" not in runtime.environment({**options, "scenario": "D"})
    assert "INFERENCE_BENCH_OUTPUT_TOKEN_CAP" not in runtime.environment({**options, "scenario_a_output_tokens": original["scenario_a_output_tokens"]})


def test_speed_baseline_runs_upstream_precompute(monkeypatch, tmp_path):
    """Drive upstream's precompute command with its torch settings, or install the shared measurement instead."""
    from inferencebench.assets.scripts import runtime

    monkeypatch.setattr(runtime, "INFERENCE", tmp_path / "inference")
    monkeypatch.setattr(runtime, "ARTIFACTS", tmp_path / "artifacts")
    commands = []
    envs = []
    monkeypatch.setattr(runtime, "run_upstream", lambda command, log, env=None, **kwargs: (commands.append(command), envs.append(env)))
    options = upstream_options(request_limit=10)
    folder = runtime.speed_baseline(options)
    assert folder == tmp_path / "inference/baselines/speed/torch/inference_scenario_a_input_heavy/mistralai_Mistral-7B-Instruct-v0.3"
    [command] = commands
    assert command[1:3] == ["-m", "src.eval.inference.precompute_baseline"]
    flags = dict(zip(command[3::2], command[4::2]))
    assert flags["--scenario-id"] == "inference_scenario_a_input_heavy" and flags["--seed"] == "1337"
    assert flags["--out-root"] == str(tmp_path / "inference/baselines/speed/torch")
    assert flags["--registry"] == str(tmp_path / "inference/baselines/speed/torch/mistralai_Mistral-7B-Instruct-v0.3.json")
    assert flags["--request-timeout-s"] == "900" and flags["--concurrency-override"] == "1" and flags["--request-limit"] == "10"
    # The baseline replays the held-out seed's arrivals, as final scoring does.
    assert envs == [{"INFERENCE_BENCH_ARRIVAL_SEED": "1337", "INFERENCE_BENCH_OUTPUT_TOKEN_CAP": "16"}]

    (tmp_path / "artifacts/cached/speed").mkdir(parents=True)
    for name in ["requests.jsonl", "baseline_metrics.json"]:
        (tmp_path / "artifacts/cached/speed" / name).write_text(name)
    runtime.speed_baseline({**options, "cached_speed_baseline": True})
    assert len(commands) == 1 and (folder / "baseline_metrics.json").read_text() == "baseline_metrics.json"


def fake_quality_runs(tmp_path, monkeypatch, outcomes):
    """Replace upstream's quality precompute with scripted generation rows, first for all questions and then per retry."""
    from inferencebench.assets.scripts import runtime

    samples = [{"sample_id": str(i), "messages": [{"role": "user", "content": str(i)}], "gold_answer": "A", "max_new_tokens": 2048, "temperature": 0} for i in range(2)]
    source = tmp_path / "samples/mmlu_pro/248_2/samples.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text("".join(json.dumps(row) + "\n" for row in samples))
    calls = []

    def run_upstream(command, log, env=None, **kwargs):
        """Write the registry and generation rows the real command would leave behind."""
        flags = dict(zip(command[3::2], command[4::2]))
        calls.append((flags, env or {}))
        out = Path(flags["--out-root"]) / "mmlu_pro" / f"{flags['--seed']}_{flags['--mmlupro-n']}"
        out.mkdir(parents=True, exist_ok=True)
        if flags["--mmlupro-n"] != "1":
            rows = [{"sample_id": str(i), "request_index": i, "gold_answer": "A", **outcome} for i, outcome in enumerate(outcomes[0])]
        else:
            selected = json.loads(Path(env["INFERENCE_BENCH_QUALITY_MMLUPRO_SAMPLES_FILE"]).read_text())
            rows = [{"sample_id": selected["sample_id"], "request_index": 0, "gold_answer": "A", **outcomes[len(calls) - 1][0]}]
        (out / "baseline_generations.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        correct = [row for row in rows if row.get("parsed_answer") == "A"]
        Path(flags["--registry"]).parent.mkdir(parents=True, exist_ok=True)
        Path(flags["--registry"]).write_text(json.dumps({"datasets": {"mmlu_pro": [{"seed": int(flags["--seed"]), "n": int(flags["--mmlupro-n"]), "accuracy": len(correct) / len(rows)}]}}))

    monkeypatch.setattr(runtime, "run_upstream", run_upstream)
    monkeypatch.setattr(runtime, "INFERENCE", tmp_path / "inference")
    monkeypatch.setattr(runtime, "ARTIFACTS", tmp_path / "artifacts")
    monkeypatch.setitem(sys.modules, "inference", SimpleNamespace(
        quality_gate=SimpleNamespace(get_quality_specs=lambda: ([SimpleNamespace(samples_file=source, seed=248, limit=2)], 0.95, None)),
        precompute_quality_baseline=SimpleNamespace(_accuracy=lambda rows: sum(row.get("parsed_answer") == row["gold_answer"] for row in rows) / len(rows)),
    ))
    return calls


@pytest.mark.parametrize("attempts,retry_success,expected", [(1, None, "accepted"), (2, True, "repaired"), (2, False, "error")])
def test_quality_reference_completion_policy(monkeypatch, tmp_path, attempts, retry_success, expected):
    """Accept upstream's registry as measured at one attempt, or retry failed questions alone and require completeness."""
    from inferencebench.assets.scripts import runtime

    first = [{"success": True, "parsed_answer": "B"}, {"success": False, "parsed_answer": None, "error": "timeout after 300s"}]
    retry = [{"success": bool(retry_success), "parsed_answer": "A" if retry_success else None}]
    calls = fake_quality_runs(tmp_path, monkeypatch, [first, retry])
    options = upstream_options(quality_samples=2, quality_baseline_max_attempts=attempts, quality_reference_backend="transformers")
    if expected == "error":
        with pytest.raises(RuntimeError, match="did not complete every request"):
            runtime.quality_reference(options)
        return
    reference = runtime.quality_reference(options)
    registry = json.loads(reference["registry"].read_text())["datasets"]["mmlu_pro"][0]
    assert reference["registry"].name == "mistralai_Mistral-7B-Instruct-v0.3_torch.json" and reference["concurrency"] == 1
    assert calls[0][0]["--backend"] == "torch" and calls[0][0]["--concurrency"] == "1" and calls[0][0]["--request-timeout-s"] == "300"
    if expected == "accepted":
        assert len(calls) == 1 and registry["accuracy"] == 0.0 and reference["complete"] is False
        return
    assert len(calls) == 2 and calls[1][0]["--concurrency"] == "1" and calls[1][0]["--mmlupro-n"] == "1"
    assert json.loads(Path(calls[1][1]["INFERENCE_BENCH_QUALITY_MMLUPRO_SAMPLES_FILE"]).read_text())["sample_id"] == "1"
    assert registry["accuracy"] == 0.5 and "retried" in registry["note"] and reference["retried_requests"] == 1
    resolved = [json.loads(line) for line in reference["generations"].with_name("resolved_generations.jsonl").read_text().splitlines()]
    assert [row["request_index"] for row in resolved] == [0, 1] and resolved[0]["parsed_answer"] == "B" and resolved[1]["success"]


@pytest.mark.parametrize("backend", ["transformers", "vllm"])
def test_prepare_orders_servers_and_records_provenance(monkeypatch, tmp_path, backend):
    """Start upstream's Transformers server for the speed baseline, measure the reference with the configured backend, and record provenance."""
    from inferencebench.assets.scripts import runtime
    from inferencebench.vendored import UPSTREAM

    calls = fake_quality_runs(tmp_path, monkeypatch, [[{"success": True, "parsed_answer": "A"}, {"success": True, "parsed_answer": "B"}]])
    events = []
    real_quality = runtime.run_upstream

    def run_upstream(command, log, env=None, **kwargs):
        """Record upstream commands and leave the speed files the precompute would write."""
        module = command[2]
        events.append(module.rsplit(".", 1)[-1])
        if module.endswith("precompute_baseline"):
            folder = Path(dict(zip(command[3::2], command[4::2]))["--out-root"]) / "inference_scenario_a_input_heavy" / "mistralai_Mistral-7B-Instruct-v0.3"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "requests.jsonl").write_text('{"messages": []}\n')
            (folder / "baseline_metrics.json").write_text(json.dumps({"baseline": measurements()}))
        elif module.endswith("precompute_quality_baseline"):
            real_quality(command, log, env=env, **kwargs)

    monkeypatch.setattr(runtime, "run_upstream", run_upstream)
    monkeypatch.setattr(runtime, "ROOT", UPSTREAM)
    monkeypatch.setattr(runtime, "TASK", tmp_path / "task")
    monkeypatch.setattr(runtime, "TRUSTED", tmp_path / "artifacts/trusted")
    monkeypatch.setattr(runtime, "BUNDLE", tmp_path / "bundle")
    monkeypatch.setattr(runtime, "PROFILE", tmp_path / "profile.d/inferencebench.sh")
    (tmp_path / "task").mkdir()
    (tmp_path / "artifacts/install").mkdir(parents=True)
    (tmp_path / "artifacts/install/upstream.lock").write_text(json.dumps({"commit": "abc", "tree": "def"}))
    (tmp_path / "inference/baselines/samples").mkdir(parents=True)
    for name in runtime.BUNDLE_FILES:
        (tmp_path / "inference" / name).write_bytes((UPSTREAM / "src/eval/inference" / name).read_bytes())
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=lambda *args, **kwargs: str(tmp_path / "snapshots/rev")))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setattr(runtime, "wait_ready", Mock(return_value=True))
    monkeypatch.setattr(runtime.subprocess, "check_output", Mock(return_value="0.19.0\n"))
    commands = []

    def popen(command, **kwargs):
        """Record each server launch as a live process."""
        events.append(("start", command[3] if command[0] == "python3" else command[1]))
        commands.append(command)
        return Mock(pid=len(events), poll=Mock(return_value=None), wait=Mock())

    monkeypatch.setattr(runtime.subprocess, "Popen", popen)
    monkeypatch.setattr(runtime.os, "killpg", Mock(side_effect=lambda pid, sig: events.append("stop")))

    options = upstream_options(quality_samples=2, quality_reference_backend=backend, baseline_dtype="bfloat16")
    registry = "mistralai_Mistral-7B-Instruct-v0.3" + ("_torch" if backend == "transformers" else "") + ".json"
    runtime.prepare(options)
    assert all(command[command.index("--dtype") + 1] == "bfloat16" for command in commands)
    assert commands[0][:4] == ["python3", "-u", "-m", "src.eval.inference.servers.transformers_openai_server"]
    transformers, vllm = ("start", "src.eval.inference.servers.transformers_openai_server"), ("start", "serve")
    if backend == "transformers":
        assert events == ["cache_samples", "cache_samples", "cache_samples", transformers, "precompute_baseline", "precompute_quality_baseline", "stop"]
        assert calls[0][0]["--concurrency"] == "1" and calls[0][0]["--backend"] == "torch"
    else:
        assert events == ["cache_samples", "cache_samples", "cache_samples", transformers, "precompute_baseline", "stop", vllm, "precompute_quality_baseline", "stop"]
        assert calls[0][0]["--concurrency"] == "4" and calls[0][0]["--backend"] == "vllm" and calls[0][0]["--registry"].endswith("/mistralai_Mistral-7B-Instruct-v0.3.json")
    trusted = tmp_path / "artifacts/trusted"
    provenance = json.loads((trusted / "provenance.json").read_text())
    assert provenance["quality_reference"] == {"backend": backend, "vllm_version": "0.19.0" if backend == "vllm" else None, "concurrency": 1 if backend == "transformers" else 4, "retried_requests": 0, "complete": True}
    assert (trusted / "quality" / registry).is_file() and (trusted / "quality/samples.jsonl").is_file()
    assert provenance["upstream"]["commit"] == "abc" and provenance["speed_baseline"] == "measured" and provenance["downloaded_model_revision"] == "rev"
    assert (trusted / "speed/requests.jsonl").is_file() and (trusted / "speed/baseline_metrics.json").is_file()
    assert (trusted / "quality/samples.jsonl").is_file() and (trusted / "quality/baseline_generations.jsonl").is_file()
    assert json.loads((trusted / "environment.json").read_text())["INFERENCE_BENCH_QUALITY_BASELINE_BACKEND"] == ("torch" if backend == "transformers" else "vllm")
    assert (tmp_path / "bundle/runner.py").is_file() and (tmp_path / "bundle/baselines/quality").is_dir() and (tmp_path / "task/evaluate.py").is_file()


@pytest.mark.parametrize(
    "dead,error,expected",
    [
        (True, "Timed out waiting for server at http://127.0.0.1:8000", "launcher"),
        (False, "Timed out waiting for server at http://127.0.0.1:8000", "unreachable"),
        (True, "broken dataset", "launcher"),
        (False, "broken dataset", "error"),
    ],
)
def test_final_evaluation_failures(local_task, monkeypatch, tmp_path, dead, error, expected):
    """Score a dead launcher or unreachable server as invalid, like upstream, while other evaluator failures stay errors."""
    from inferencebench.assets.scripts import runtime

    task, env = local_task
    process = Mock(returncode=1 if dead else None)
    process.poll.return_value = 1 if dead else None
    monkeypatch.setattr(runtime, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(runtime, "restore_trusted", Mock())
    monkeypatch.setattr(runtime.shutil, "copy", Mock())
    monkeypatch.setattr(runtime.subprocess, "Popen", Mock(return_value=process))
    monkeypatch.setattr(runtime.os, "killpg", Mock())
    monkeypatch.setattr(runtime, "wait_ready", Mock(return_value=True))
    monkeypatch.setattr(runtime, "evaluate", Mock(return_value={"scenario": "A", "model_id": "m", "profiles": {}, "vram_peak_mb": 0.0, "error": error}))

    async def execute(*args, **kwargs):
        """Drive the actual runtime failure path through Inspect's normal scorer."""
        runtime.final(task.dataset[0].metadata)
        return ExecResult(success=True, returncode=0, stdout="", stderr="")

    async def read_file(path):
        """Return the runtime's recorded failure evidence to the real scorer."""
        return (tmp_path / path.rsplit("/", 1)[-1]).read_text() if path.endswith("final.json") else ""

    env.exec.side_effect = execute
    env.read_file = read_file
    [log] = inspect_eval(task, solver=generate(), model="mockllm/subject", display="none", log_dir="logs")
    if expected == "error":
        assert log.status == "error" and "broken dataset" in log.samples[0].error.message
        assert not log.samples[0].scores
        return
    assert log.status == "success", log.error
    score = log.samples[0].scores["inference_speedup"]
    assert score.value == {"speedup": 1.0}
    assert score.metadata["final"]["evaluator_error"] == error
    if expected == "launcher":
        assert score.explanation == "Canonical launcher exited during final evaluation" and score.metadata["final"]["launcher_returncode"] == 1
    else:
        assert score.explanation == error


def test_final_evaluation_uses_upstream_command_and_retries(monkeypatch, tmp_path):
    """Run upstream's task evaluate.py against the restored requests, retrying on its schedule until metrics exist."""
    from inferencebench.assets.scripts import runtime
    from inferencebench.vendored import UPSTREAM

    monkeypatch.setattr(runtime, "ROOT", UPSTREAM)
    monkeypatch.setattr(runtime, "INFERENCE", tmp_path / "inference")
    monkeypatch.setattr(runtime, "ARTIFACTS", tmp_path / "artifacts")
    attempts = []

    def run_upstream(command, log, env=None, timeout=None, check=True):
        """Fail twice, then write metrics on the third attempt."""
        attempts.append((command, env, timeout, check))
        if len(attempts) == 3:
            Path(command[command.index("--json-output-file") + 1]).write_text(json.dumps(measurements()))
        return 1

    monkeypatch.setattr(runtime, "run_upstream", run_upstream)
    options = upstream_options(request_limit=10)
    assert runtime.evaluate(options) == measurements()
    assert len(attempts) == 3
    command, env, timeout, check = attempts[0]
    assert command[1] == str(UPSTREAM / "src/eval/tasks/inference_scenario_a_input_heavy/evaluate.py")
    flags = dict(zip(command[2::2], command[3::2]))
    assert flags["--requests-file"] == str(tmp_path / "inference/baselines/speed/torch/inference_scenario_a_input_heavy/mistralai_Mistral-7B-Instruct-v0.3/requests.jsonl")
    assert flags["--quality-tau"] == "0.95" and flags["--request-limit"] == "10" and "--request-timeout-s" not in flags
    assert env == {"INFERENCE_BENCH_DATASET_SEED": "1337", "INFERENCE_BENCH_ARRIVAL_SEED": "1337", "INFERENCE_BENCH_OUTPUT_TOKEN_CAP": "16"} and timeout == 3600 and check is False
    attempts.clear()
    monkeypatch.setattr(runtime, "run_upstream", lambda command, log, **kwargs: attempts.append(command) and 1)
    assert runtime.evaluate(options) is None
    assert len(attempts) == 5 and [("--request-timeout-s" in command) for command in attempts] == [False, False, False, True, True]
    assert attempts[3][attempts[3].index("--request-timeout-s") + 1] == "150"


def prepare_archive(rows, provenance=True):
    """Build the archive the sandbox returns after preparation, with the trusted files the host keeps."""
    import io
    import tarfile

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        def add(name, text):
            """Add one text file under trusted/."""
            data = text.encode()
            info = tarfile.TarInfo(f"trusted/{name}")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        add("speed/requests.jsonl", rows)
        add("speed/baseline_metrics.json", json.dumps({"baseline": measurements()}))
        add("speed/baseline_generations.jsonl", "{}\n")
        if provenance:
            add("quality/samples.jsonl", "{}\n")
            add("quality/baseline_generations.jsonl", "{}\n")
            add("quality/mistralai_Mistral-7B-Instruct-v0.3_torch.json", json.dumps({"datasets": {"mmlu_pro": [{"seed": 248, "n": 500, "accuracy": 0.302}]}}))
            add("provenance.json", json.dumps({"downloaded_model_revision": "rev", "speed_baseline": "measured", "quality_reference": {"backend": "transformers", "vllm_version": None, "concurrency": 1, "retried_requests": 0, "complete": True}}))
            add("environment.json", json.dumps({"INFERENCE_BENCH_BASE_MODEL": "mistralai/Mistral-7B-Instruct-v0.3"}))
    return buffer.getvalue()


def host_environment(tmp_path, prepare_outcomes):
    """Fake a GPU sandbox for the host preparation solver, scripting each preparation's success and archive."""
    written, uploads, prepares = {}, [], []

    async def execute(command, **kwargs):
        """Answer the GPU inventory, record installs and preparations, and script preparation outcomes."""
        if command[0] == "nvidia-smi":
            return ExecResult(success=True, returncode=0, stdout="name, memory.total [MiB], driver_version\nNVIDIA H100 80GB HBM3, 81559 MiB, 580.95.05\n", stderr="")
        if "prepare" in command:
            prepares.append(json.loads(written["options"]))
            outcome = prepare_outcomes.pop(0)
            return ExecResult(success=outcome == "pass", returncode=0 if outcome == "pass" else 1, stdout="", stderr="reference failed")
        return ExecResult(success=True, returncode=0, stdout="", stderr="")

    async def read_file(path):
        """Serve the final measurement and the fake judge's verdicts to the scorer."""
        if path.endswith("final.json"):
            return json.dumps(measurements(2))
        return verdict_file(path) if path.endswith("_judgement.txt") else ""

    async def write_file(path, content):
        """Capture the options handed to the sandbox runtime."""
        if path.endswith("options.json"):
            written["options"] = content
        if path.endswith("upstream.tar"):
            written["upstream"] = len(content)

    async def upload(local, remote):
        """Record which shared files reach the sandbox, by cache kind."""
        uploads.append(remote.removeprefix("/tmp/inferencebench/cached/"))

    async def download(remote, local):
        """Return the preparation archive, with provenance only after a successful preparation."""
        Path(local).write_bytes(prepare_archive('{"messages": []}\n', provenance=written.get("last_prepare_passed", True)))

    env = SimpleNamespace(resource_id="pod", exec=execute, read_file=read_file, write_file=write_file, upload=upload, download=download, terminate=AsyncMock())
    return env, written, uploads, prepares


def test_shared_measurements_reuse(monkeypatch, tmp_path):
    """Measure the speed baseline once per workload and GPU model, install the upstream copy each time, and share the stored files."""
    environment = importlib.import_module("inferencebench.environment")
    monkeypatch.setattr(environment, "BASELINE_CACHE", tmp_path / "baselines")
    env, written, uploads, prepares = host_environment(tmp_path, ["pass", "pass", "pass", "pass"])
    monkeypatch.setattr(environment, "gpu_environment", lambda: env)
    monkeypatch.setattr(SCORERS, "restart_for_scoring", AsyncMock(return_value=env))

    def run(**overrides):
        """Run one sample through the real preparation solver and return its metadata."""
        task = inference_bench(**{"scenarios": "A", "seed_pairs": [[21, 1337]], "agent_seconds": 2, "quality_reference_backend": "transformers", **overrides})
        task.sandbox = None
        [log] = inspect_eval(task, solver=generate(), model="mockllm/subject", model_roles={"integrity": judge_model()}, display="none", log_dir="logs")
        assert log.status == "success", log.error
        return log.samples[0].metadata

    first = run()
    assert written["upstream"] > 100_000
    assert prepares[-1]["cached_speed_baseline"] is False and first["speed_baseline"]["source"] == "measured"
    assert first["provenance"]["downloaded_model_revision"] == "rev"
    [speed] = list((tmp_path / "baselines").iterdir())
    manifest = json.loads((speed / "manifest.json").read_text())
    assert manifest["identity"]["gpu"] == "NVIDIA H100 80GB HBM3" and manifest["identity"]["scenario"] == "A" and manifest["identity"]["eval_seed"] == 1337
    assert manifest["identity"]["upstream_commit"].startswith("24cdf88") and manifest["identity"]["patches"]
    assert all((speed / name).exists() for name in ["requests.jsonl", "baseline_metrics.json", "baseline_generations.jsonl"])
    assert speed.name.startswith("A-seed1337") and uploads == []

    second = run()
    assert prepares[-1]["cached_speed_baseline"] is True and second["speed_baseline"]["source"] == "cache"
    assert sorted(uploads) == ["speed/baseline_metrics.json", "speed/requests.jsonl"]
    assert second["speed_baseline"]["folder"] == str(speed.resolve())

    run(seed_pairs=[[21, 428]])
    assert prepares[-1]["cached_speed_baseline"] is False and len(list((tmp_path / "baselines").iterdir())) == 2


def test_prepare_failure_retains_measured_speed_baseline(monkeypatch, tmp_path):
    """Store a completed speed measurement when the reference fails afterwards, so the retry skips it."""
    environment = importlib.import_module("inferencebench.environment")
    monkeypatch.setattr(environment, "BASELINE_CACHE", tmp_path / "baselines")
    env, written, uploads, prepares = host_environment(tmp_path, ["fail", "pass"])
    original_execute = env.exec

    async def execute(command, **kwargs):
        """Mark whether the last preparation passed so the archive omits provenance after a failure."""
        result = await original_execute(command, **kwargs)
        if "prepare" in command:
            written["last_prepare_passed"] = result.success
        return result

    env.exec = execute
    monkeypatch.setattr(environment, "gpu_environment", lambda: env)
    monkeypatch.setattr(SCORERS, "restart_for_scoring", AsyncMock(return_value=env))

    def run():
        """Run one sample through the real preparation solver."""
        task = inference_bench(scenarios="A", seed_pairs=[[21, 1337]], agent_seconds=2)
        task.sandbox = None
        [log] = inspect_eval(task, solver=generate(), model="mockllm/subject", model_roles={"integrity": judge_model()}, display="none", log_dir="logs")
        return log

    failed = run()
    assert failed.status == "error" and "reference failed" in failed.samples[0].error.message
    [stored] = list((tmp_path / "baselines").iterdir())
    assert stored.name.startswith("A-seed1337") and (stored / "requests.jsonl").exists() and (stored / "baseline_metrics.json").exists()
    passed = run()
    assert passed.status == "success", passed.error
    assert prepares[-1]["cached_speed_baseline"] is True
    assert sorted(uploads) == ["speed/baseline_metrics.json", "speed/requests.jsonl"]
    assert passed.samples[0].metadata["speed_baseline"]["source"] == "cache" and len(list((tmp_path / "baselines").iterdir())) == 1


def test_speed_baseline_identity_separates_configurations():
    """Key the shared speed baseline by the workload settings that differ between configurations."""
    from inferencebench.environment import speed_baseline_identity

    default = inference_bench(scenarios="A", seed_pairs=[[21, 1337]]).dataset[0].metadata
    original = inference_bench(**{**load_config("run_configs/original.yaml")["task"]["args"], "scenarios": "A", "seed_pairs": [[21, 1337]]}).dataset[0].metadata
    identity = speed_baseline_identity(default, "NVIDIA H100 80GB HBM3")
    assert identity != speed_baseline_identity(original, "NVIDIA H100 80GB HBM3")
    assert identity != speed_baseline_identity(default, "NVIDIA A100 80GB PCIe")
