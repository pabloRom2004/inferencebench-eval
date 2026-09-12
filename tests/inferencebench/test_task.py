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

from inferencebench import inference_bench, original_agent, react_agent
from inferencebench.dataset import SCENARIOS, cached_requests, load_request_cache
from inferencebench.metrics import aggregate_speedup, complete_mean, performance
from inferencebench.prompts import PROMPTS
from inferencebench.run_config import load_config
from inferencebench.scorers import parse_judgment
from inferencebench.tools import web_search

TASK = importlib.import_module("inferencebench.task")
SCORERS = importlib.import_module("inferencebench.scorers")
HARNESS = importlib.import_module("inferencebench.harness_original")
TOOLS = importlib.import_module("inferencebench.tools")


@pytest.fixture(autouse=True)
def fresh_quality_reference(monkeypatch):
    """Keep existing evaluator tests on their controlled fresh-reference path; cache reuse is tested separately."""
    monkeypatch.setattr(TASK, "load_quality_cache", lambda options: None)


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
            (tmp_path / "baseline.json").write_text(json.dumps(measurements()))
            state.metadata["prepared"] = True
            return state

        return solve

    async def read_file(path):
        """Return only the freshly generated final server outputs expected by the scorer."""
        return (
            json.dumps(measurements(2))
            if path.endswith("final.json")
            else "Mistral server log"
        )

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
    assert task.dataset[0].metadata["agent_seconds"] is None
    assert "no wall-clock optimization limit" in task.dataset[0].input
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
        "request_cache",
        "quality_baseline_max_attempts",
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
    assert config["task"]["args"]["scorer"]["args"]["include_transcript"] is True
    assert original["task"]["args"]["scorer"]["args"] == {
        **config["task"]["args"]["scorer"]["args"],
        "include_transcript": False,
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


@pytest.mark.parametrize("configuration", ["default", "original"])
def test_prepared_request_prefixes(configuration):
    """Load complete configured workloads for every scenario and seed while preserving shared prefixes."""
    config = load_config(f"run_configs/{configuration}.yaml")["task"]["args"]
    task = inference_bench(**config)
    cache = load_request_cache(task.dataset[0].metadata)
    prefixes = load_request_cache(inference_bench().dataset[0].metadata)
    seeds = load_config("run_configs/original.yaml")["task"]["args"]["seed_pairs"]
    for scenario, record in SCENARIOS.items():
        lengths = record["config"]["synthetic"]
        for seed in {seed for pair in seeds for seed in pair}:
            rows = cached_requests(cache, scenario, seed, config["request_limit"])
            assert len(rows) == (10 if configuration == "default" else record["config"]["num_requests"])
            assert rows[:10] == cached_requests(prefixes, scenario, seed, 10)
            assert cached_requests(cache, scenario, seed, 1) == rows[:1]
            with pytest.raises(ValueError, match="requested"):
                cached_requests(cache, scenario, seed, len(rows) + 1)
            for row in rows:
                assert row["messages"]
                assert row["ignore_eos"] is True
                assert row["temperature"] == record["config"]["temperature"]
                assert 0.8 * lengths["input_len"] <= row["target_input_token_count"] <= lengths["input_len"]
                assert row["input_token_count"] <= row["target_input_token_count"]
                assert 0.8 * lengths["output_len"] <= row["max_new_tokens"] <= lengths["output_len"]
                assert len(row["content_hash"]) == 64


@pytest.mark.parametrize("options", [
    {"base_model": "another/model"},
    {"max_model_len": 4096},
    {"seed_pairs": [[123, 456]]},
    {"request_limit": 11},
    {"request_limit": None},
])
def test_incompatible_request_cache(options):
    """Reject unsupported cached workloads before GPU allocation while preserving explicit upstream sampling."""
    with pytest.raises(ValueError, match="request_cache: null"):
        inference_bench(**options, quality_cache=None)
    assert inference_bench(**options, request_cache=None, quality_cache=None).dataset


def test_request_cache_checksum(tmp_path):
    """Detect modified prepared data before any baseline or subject execution."""
    import gzip

    options = inference_bench().dataset[0].metadata
    cache = load_request_cache(options)
    cache["requests"]["A"]["21"][0]["max_new_tokens"] += 1
    path = tmp_path / "requests.json.gz"
    path.write_bytes(gzip.compress(json.dumps(cache).encode()))
    with pytest.raises(ValueError, match="requests_sha256"):
        inference_bench(request_cache=str(path))


@pytest.mark.parametrize("use_cache", [True, False])
def test_runtime_request_selection(monkeypatch, tmp_path, use_cache):
    """Keep cached preparation off the corpus sampler and retain the original path when disabled."""
    from inferencebench.assets.scripts import runtime

    options = inference_bench(request_cache=None).dataset[0].metadata
    options["request_cache"] = "prepared" if use_cache else None
    rows = [{"messages": [{"role": "user", "content": "prepared request"}]}]
    runner = SimpleNamespace(
        _get_tokenizer=Mock(return_value="tokenizer"),
        _load_requests_jsonl=Mock(return_value=(rows, [])),
        _prepare_requests=Mock(return_value=(rows, [])),
    )
    monkeypatch.setitem(sys.modules, "inference", SimpleNamespace(runner=runner))
    monkeypatch.setattr(runtime, "ARTIFACTS", tmp_path)
    config = SCENARIOS["A"]["config"]
    assert runtime.prepare_requests(options, config, "dev") == rows
    if use_cache:
        runner._load_requests_jsonl.assert_called_once_with(
            tmp_path / "dev-requests.jsonl", config, 10, "tokenizer", 32768
        )
        runner._prepare_requests.assert_not_called()
    else:
        runner._prepare_requests.assert_called_once_with(config, 10, "tokenizer", 32768)
        runner._load_requests_jsonl.assert_not_called()


@pytest.mark.parametrize("use_cache", [True, False])
def test_development_wrapper_request_selection(monkeypatch, tmp_path, use_cache):
    """Run the generated development command and verify its cached or original dataset selection."""
    import argparse
    import os
    import runpy
    from unittest.mock import patch

    from inferencebench.assets.scripts import runtime

    context = tmp_path / "src/eval/tasks/_shared/task_context"
    context.mkdir(parents=True)
    for name in ["start_server.sh", "test_server.sh"]:
        (context / name).touch()
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    monkeypatch.setattr(runtime, "TASK", tmp_path)
    monkeypatch.setattr(sys, "argv", ["evaluate.py"])
    observed = []

    def run_evaluation(task, args):
        """Observe the same environment-based file selection used by the upstream runner."""
        observed.append(os.environ.get("INFERENCE_BENCH_REQUESTS_FILE"))

    runner = SimpleNamespace(build_parser=argparse.ArgumentParser, run_evaluation=run_evaluation)
    monkeypatch.setitem(sys.modules, "inference.runner", runner)
    options = inference_bench(request_cache=None).dataset[0].metadata
    options["request_cache"] = "prepared" if use_cache else None
    with patch.dict(os.environ, {"HF_HOME": str(tmp_path)}, clear=True):
        runtime.install_workspace(options)
        runpy.run_path(str(tmp_path / "evaluate.py"))
    assert observed == [str(tmp_path / "requests.jsonl") if use_cache else None]


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
    monkeypatch.setattr(importlib.import_module("inferencebench.cli"), "sandbox", lambda: env)
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
        name = f"inspect_swe/{harness}"
        args = {"cwd": "/home/agent/task", "user": "root", "version": "auto"}
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
            *[part for key, value in args.items() for part in ("-S", f"{key}={value}")],
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
        max_model_len=16384, request_cache=None, quality_cache=None, quality_concurrency=2, quality_samples=16, quality_seed=0
    )
    import os
    from unittest.mock import patch

    with patch.dict(os.environ, {"HF_HOME": "/tmp/cache"}):
        env = environment(task.dataset[0].metadata)
    assert env["INFERENCE_BENCH_MAX_MODEL_LEN"] == "16384"
    assert env["INFERENCE_BENCH_QUALITY_CONCURRENCY"] == "2"
    assert env["INFERENCE_BENCH_QUALITY_MMLUPRO_N"] == "16"
    assert env["INFERENCE_BENCH_QUALITY_SEED"] == "0"


@pytest.mark.parametrize(
    "successes,accuracy,valid",
    [
        ([True, True], 0.5, True),
        ([False, False], 0.0, False),
        ([True, False], 0.5, False),
        ([True], 1.0, False),
        ([True, True], 0.0, False),
        ([True, True], math.nan, False),
        ([True, True], math.inf, False),
    ],
)
def test_quality_baseline_failure_accounting(
    monkeypatch, tmp_path, successes, accuracy, valid
):
    """Reject failed or undefined references before model generation without assigning a submission score."""
    from inferencebench.assets.scripts import runtime

    environment = importlib.import_module("inferencebench.environment")
    spec = SimpleNamespace(seed=248, limit=2)
    log_path = tmp_path / "baseline_generations.jsonl"
    log_path.write_text("".join(json.dumps({"success": ok}) + "\n" for ok in successes))
    upstream = SimpleNamespace(
        _run_dataset=Mock(return_value=(accuracy, log_path, None))
    )
    monkeypatch.setitem(
        sys.modules, "inference", SimpleNamespace(precompute_quality_baseline=upstream)
    )
    monkeypatch.setattr(runtime, "ARTIFACTS", tmp_path)
    task = inference_bench(scenarios="A", seed_pairs=[[21, 1337]], quality_cache=None)
    options = task.dataset[0].metadata
    options["quality_baseline_max_attempts"] = 1
    if valid:
        runtime.prepare_quality_baseline(options, spec)
        registry = json.loads((tmp_path / "quality.json").read_text())
        assert registry["datasets"]["mmlu_pro"][0]["accuracy"] == accuracy
        return

    async def execute(command, **kwargs):
        """Run reference validation at the real setup command boundary with simulated GPU results."""
        if "prepare" in command:
            try:
                runtime.prepare_quality_baseline(options, spec)
            except RuntimeError as error:
                return ExecResult(success=False, returncode=1, stdout="", stderr=str(error))
        return ExecResult(success=True, returncode=0, stdout="", stderr="")

    env = SimpleNamespace(resource_id="test", exec=execute, write_file=AsyncMock())
    monkeypatch.setattr(environment, "gpu_environment", lambda: env)
    task.sandbox = None
    [log] = inspect_eval(
        task, model="mockllm/subject", display="none", log_dir="logs"
    )
    assert log.status == "error"
    assert "Transformers quality baseline" in log.samples[0].error.message
    assert not log.samples[0].scores
    assert not log.stats.model_usage
    assert not (tmp_path / "quality.json").exists()


@pytest.mark.parametrize("retry_success", [False, True])
def test_quality_reference_retry(monkeypatch, tmp_path, retry_success):
    """Keep successful answers, retry only failed IDs with unchanged inputs, and stop at the attempt bound."""
    from dataclasses import dataclass

    from inferencebench.assets.scripts import runtime

    @dataclass
    class Spec:
        """Use the upstream dataset specification's replaceable fields."""
        samples_file: Path
        seed: int
        limit: int

    samples = [{"sample_id": str(i), "messages": [{"role": "user", "content": str(i)}],
                "gold_answer": "A", "max_new_tokens": 2048, "temperature": 0} for i in range(2)]
    source = tmp_path / "quality-samples.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in samples))
    spec = Spec(source, 248, 2)
    calls = []

    def run_dataset(selection, url, model, timeout, concurrency, out):
        """Return a completed wrong answer and one transport failure, then replay the failed input."""
        calls.append(selection)
        assert timeout == 300
        if len(calls) == 1:
            assert concurrency == 4 and selection.limit == 2
            rows = [{"sample_id": "0", "request_index": 0, "success": True, "gold_answer": "A", "parsed_answer": "B"},
                    {"sample_id": "1", "request_index": 1, "success": False, "gold_answer": "A", "parsed_answer": None}]
        else:
            assert concurrency == 1 and selection.limit == 1
            assert json.loads(selection.samples_file.read_text()) == samples[1]
            rows = [{"sample_id": "1", "request_index": 0, "success": retry_success, "gold_answer": "A", "parsed_answer": "A" if retry_success else None}]
        path = out / "baseline_generations.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return 0.0, path, None

    def accuracy(rows):
        """Mirror the upstream all-question denominator for this deterministic fixture."""
        return sum(row["parsed_answer"] == row["gold_answer"] for row in rows) / len(rows)

    monkeypatch.setitem(sys.modules, "inference", SimpleNamespace(precompute_quality_baseline=SimpleNamespace(_run_dataset=run_dataset, _accuracy=accuracy)))
    monkeypatch.setattr(runtime, "ARTIFACTS", tmp_path)
    options = inference_bench().dataset[0].metadata
    if retry_success:
        runtime.prepare_quality_baseline(options, spec)
        registry = json.loads((tmp_path / "quality.json").read_text())
        assert registry["datasets"]["mmlu_pro"][0]["accuracy"] == 0.5
        resolved = [json.loads(line) for line in (tmp_path / "quality-baseline/resolved_generations.jsonl").read_text().splitlines()]
        assert [row["request_index"] for row in resolved] == [0, 1]
        assert resolved[0]["parsed_answer"] == "B"
    else:
        with pytest.raises(RuntimeError, match="did not complete every request"):
            runtime.prepare_quality_baseline(options, spec)
        assert not (tmp_path / "quality.json").exists()
    assert len(calls) == 2
    initial = [json.loads(line) for line in (tmp_path / "quality-baseline/baseline_generations.jsonl").read_text().splitlines()]
    assert initial[1]["success"] is False
    assert (tmp_path / "quality-baseline/attempt-2/1/baseline_generations.jsonl").exists()


@pytest.mark.parametrize(
    "dead,error,expected",
    [
        (
            True,
            TimeoutError("Timed out waiting for server at http://127.0.0.1:8000"),
            "success",
        ),
        (
            False,
            TimeoutError("Timed out waiting for server at http://127.0.0.1:8000"),
            "error",
        ),
        (True, ValueError("broken dataset"), "error"),
        (True, TimeoutError("unrelated evaluator timeout"), "error"),
    ],
)
def test_launcher_exit_during_evaluation(
    local_task, monkeypatch, tmp_path, dead, error, expected
):
    """Grade confirmed launcher death as invalid while preserving unrelated evaluator failures as errors."""
    from inferencebench.assets.scripts import runtime

    task, env = local_task
    process = Mock(returncode=1 if dead else None)
    process.poll.return_value = 1 if dead else None
    monkeypatch.setattr(runtime, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(runtime.shutil, "copy", Mock())
    monkeypatch.setattr(runtime.subprocess, "Popen", Mock(return_value=process))
    monkeypatch.setattr(runtime.os, "killpg", Mock())
    monkeypatch.setattr(runtime, "wait_ready", Mock(return_value=True))
    monkeypatch.setattr(runtime, "evaluate", Mock(side_effect=error))

    async def execute(*args, **kwargs):
        """Drive the actual runtime failure path through Inspect's normal scorer."""
        runtime.final(task.dataset[0].metadata)
        return ExecResult(success=True, returncode=0, stdout="", stderr="")

    async def read_file(path):
        """Return the runtime's recorded failure evidence to the real scorer."""
        return (tmp_path / path.rsplit("/", 1)[-1]).read_text()

    env.exec.side_effect = execute
    env.read_file = read_file
    [log] = inspect_eval(
        task,
        solver=generate(),
        model="mockllm/subject",
        display="none",
        log_dir="logs",
    )
    assert log.status == expected
    if expected == "success":
        score = log.samples[0].scores["inference_speedup"]
        assert score.value == {"speedup": 1.0}
        assert score.metadata["final"]["launcher_returncode"] == 1
        assert (
            "Timed out waiting for server" in score.metadata["final"]["evaluator_error"]
        )
    else:
        assert log.samples[0].error
        assert not log.samples[0].scores


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
    assert PROMPTS["inspect_judge_adapter"].output_format


async def test_modal_filesystem_adapter():
    """Use the supported Modal filesystem methods for writes, reads, and parent creation."""
    from inferencebench.modal_sandbox import InferenceSandbox

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
        return json.dumps(metrics) if path.endswith("final.json") else "server log"

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
    }


@pytest.mark.parametrize("legacy_strict", [True, False])
def test_openrouter_judge_optional_tool_schema(local_task, legacy_strict):
    """Reproduce the strict-schema rejection through the real provider and exercise the configured repair."""
    import httpx2
    from inspect_ai.model import GenerateConfig

    task, env = local_task
    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        function = body["tools"][0]["function"]
        assert function["name"] == "inspect_submission"
        parameters = function["parameters"]
        if function.get("strict") and set(parameters["required"]) != set(parameters["properties"]):
            return httpx2.Response(400, json={"error": {
                "message": "Invalid schema: required must include every key in properties; missing start_line",
                "type": "invalid_request_error", "code": "invalid_function_parameters",
            }})
        message = (
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "read-launcher", "type": "function", "function": {
                    "name": "inspect_submission",
                    "arguments": json.dumps({"path": "/home/agent/task/start_server.sh"}),
                },
            }]}
            if len(requests) == 1
            else {"role": "assistant", "content": "no contamination detected\nonly allowed use detected"}
        )
        return httpx2.Response(200, json={
            "id": "judge-schema-test", "object": "chat.completion", "created": 0,
            "model": "openai/gpt-6-astra",
            "choices": [{"index": 0, "message": message,
                         "finish_reason": "tool_calls" if len(requests) == 1 else "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    spec = load_config()["model_roles"]["integrity"]
    args = {} if legacy_strict else spec["args"]
    judge = get_model(
        spec["model"], api_key="local-test-no-credential",
        base_url="https://schema-test.invalid/v1",
        config=GenerateConfig(**{**spec["config"], "max_retries": 0}),
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(respond)),
        **args,
    )
    [log] = inspect_eval(
        task, solver=generate(), model="mockllm/subject",
        model_roles={"integrity": judge}, display="none", log_dir="logs",
    )
    env.terminate.assert_awaited_once()
    if legacy_strict:
        assert log.status == "error"
        assert "missing start_line" in log.samples[0].error.message
        assert not log.samples[0].scores
    else:
        assert log.status == "success", log.error
        assert len(requests) == 2
        env.exec.assert_any_await(["sed", "-n", "1,200p", "--", "/home/agent/task/start_server.sh"], timeout=30)
        assert log.samples[0].scores["inference_speedup"].value["speedup"] == 2


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
    [("default", None), ("original", None), ("default", False), ("original", True)],
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
    enabled = args["scorer"]["args"]["include_transcript"]
    task = inference_bench(**{**args, "scenarios": "A", "seed_pairs": [[21, 1337]]})
    task.sandbox = None

    def local_path(remote):
        """Map sandbox evidence paths into this test's isolated filesystem."""
        return tmp_path / "sandbox" / remote.lstrip("/")

    async def write_file(path, content):
        """Persist the real scorer's exported evidence for inspection by the mock judge."""
        path = local_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

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

    for name in ["quality.json", "heldout-requests.jsonl", "quality-samples.jsonl"]:
        (tmp_path / name).write_text("{}")
    transcript = f"{environment.REMOTE}/agent-transcript.json"
    local_path(transcript).parent.mkdir(parents=True)
    local_path(transcript).write_text("stale transcript from a previous snapshot")
    launcher = "/home/agent/task/start_server.sh"
    local_path(launcher).parent.mkdir(parents=True)
    local_path(launcher).write_text("launcher evidence")
    env.resource_id = "local-scoring-sandbox"
    env.restart = AsyncMock(return_value=env)
    env.upload = AsyncMock(side_effect=upload)
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
            ModelOutput.for_tool_call(
                "mockllm/transcript-judge", "inspect_submission", {"path": path}
            )
            for path in [launcher, transcript]
        ]
        + [
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
    assert ("agent-transcript.json" in initial.input[0].text) is enabled
    assert launcher in initial.input[0].text
    reads = [
        event
        for event in sample.events
        if event.event == "tool" and event.function == "inspect_submission"
    ]
    assert reads[0].result == "launcher evidence"
    assert ("subject transcript evidence" in reads[1].result) is enabled
    assert ("CLI tool result evidence" in reads[1].result) is enabled
    assert local_path(transcript).exists() is enabled
    if enabled:
        evidence = json.loads(local_path(transcript).read_text())
        assert evidence["messages"] == [
            message.model_dump(mode="json") for message in sample.messages
        ]
        assert "subject transcript evidence" in json.dumps(evidence["events"])
        assert json.dumps(evidence["events"]).count("CLI tool result evidence") == 1
    else:
        assert "No such file" in reads[1].result
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
        assert fs.exists(f"{path}/baseline.json")
        assert fs.exists(f"{path}/final.json")
        with tarfile.open(fileobj=io.BytesIO(fs.cat(f"{path}/submission.tar.gz"))) as archive:
            assert archive.extractfile("task/start_server.sh").read() == b"launcher evidence"
        assert fs.exists(f"{path}/agent-transcript.json") is enabled
        fs.rm(path, recursive=True)


@pytest.mark.parametrize("value", [None, "false", 0])
def test_invalid_transcript_toggle(value):
    """Reject ambiguous transcript settings before creating a scoring sandbox."""
    args = load_config()["task"]["args"]["scorer"]["args"]
    with pytest.raises(ValueError, match="include_transcript must be a boolean"):
        SCORERS.inference_speedup(**{**args, "include_transcript": value})


def test_failed_solver_retains_submission(local_task, monkeypatch, tmp_path):
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
            raise RuntimeError("native agent died")
        return solve

    [log] = inspect_eval(task, solver=broken_agent(), model="mockllm/subject",
                        display="none", log_dir="logs")
    assert log.status == "error"
    sample = read_eval_log(log.location).samples[0]
    assert "native agent died" in sample.error.message
    assert not sample.scores
    fs, path = environment.url_to_fs(f"{destination}/artifacts/{sample.uuid}")
    with tarfile.open(fileobj=io.BytesIO(fs.cat(f"{path}/submission.tar.gz"))) as archive:
        assert archive.extractfile("task/start_server.sh").read() == b"unfinished launcher"
    assert json.loads(fs.cat(f"{path}/agent-transcript.json"))["events"]
    fs.rm(path, recursive=True)
