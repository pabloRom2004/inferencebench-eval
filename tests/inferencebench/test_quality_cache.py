import importlib
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.log import read_eval_log
from inspect_ai.solver import generate
from inspect_ai.util import ExecResult

from inferencebench import inference_bench
from inferencebench.assets.scripts import runtime
from inferencebench.dataset import SCENARIOS
from inferencebench.prepare_quality import measure_reference, publish_reference
from inferencebench.quality_cache import (
    load_quality_cache,
    local_quality_folder,
    quality_identity,
    read_reference,
)
from inferencebench.run_config import load_config


@pytest.fixture
def reference(tmp_path):
    """Publish two controlled upstream answers as a complete isolated cache and Inspect log."""
    options = {**load_config()["task"]["args"], "quality_samples": 2, "quality_cache_dir": str(tmp_path / "cache")}
    folder = tmp_path / "measurement"
    folder.mkdir()
    samples = [{"sample_id": str(i), "messages": [{"role": "user", "content": f"Question {i}"}],
                "gold_answer": "A", "temperature": 0.0, "max_new_tokens": 2048} for i in range(2)]
    results = [{"sample_id": str(i), "gold_answer": "A", "parsed_answer": "A" if i == 0 else "B",
                "model_output": "A" if i == 0 else "B", "success": True} for i in range(2)]
    for name, rows in [("quality-samples.jsonl", samples), ("resolved_generations.jsonl", results)]:
        (folder / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
    (folder / "quality.json").write_text(json.dumps({"datasets": {"mmlu_pro": [{"seed": 248, "n": 2, "accuracy": 0.5}]}}))
    (folder / "provenance.json").write_text(json.dumps({"downloaded_model_revision": "fixture-revision",
        "started_at_unix": 1700000000, "completed_at_unix": 1700000010,
        "upstream_revision": quality_identity(options)["upstream_revision"]}))
    (folder / "quality-baseline.tar.gz").write_bytes(b"test archive")
    destination = local_quality_folder(options)
    log = publish_reference(folder, options, destination, Path("logs").resolve())
    yield options, destination, folder, log
    log.unlink(missing_ok=True)


def test_reference_round_trip_and_identity(reference):
    """Reuse an exact reference and retain every question, wrong answer, and measured accuracy in Inspect."""
    options, destination, _, path = reference
    assert load_quality_cache(options) == destination
    log = read_eval_log(path)
    assert log.status == "success" and log.eval.model == options["base_model"]
    assert len(log.samples) == 2
    assert [sample.scores["mmlu_pro"].value for sample in log.samples] == [1, 0]
    assert log.samples[1].output.completion == "B"
    assert log.results.scores[0].metrics["accuracy"].value == 0.5
    for change in [{"quality_seed": 42}, {"base_model": "other/model"}, {"quality_samples": 1}, {"max_model_len": 4096}]:
        with pytest.raises(ValueError, match="Run these questions first"):
            load_quality_cache({**options, **change})
    assert load_quality_cache({**options, "quality_cache": None}) is None


def test_corrupt_cache_and_failed_publication(reference, tmp_path):
    """Reject changed payloads, partial answers, and zero references without publishing a usable folder."""
    options, destination, source, _ = reference
    (destination / "quality.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_quality_cache(options)
    results = [json.loads(line) for line in (source / "resolved_generations.jsonl").read_text().splitlines()]
    for rows in [results[:1], [results[0], results[0]], [{**row, "success": False} for row in results],
                 [{**row, "parsed_answer": "B"} for row in results]]:
        (source / "resolved_generations.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        with pytest.raises(ValueError):
            publish_reference(source, options, tmp_path / "unpublished", Path("logs").resolve())
        assert not (tmp_path / "unpublished").exists()


def test_missing_reference_prevents_allocation(reference, monkeypatch):
    """Stop a full task with actionable instructions before sandbox or subject construction."""
    options, _, _, _ = reference
    module = importlib.import_module("inferencebench.task")
    monkeypatch.setattr(module, "prepare_environment", Mock(side_effect=AssertionError("setup must not start")))
    with pytest.raises(ValueError, match="inferencebench.prepare_quality"):
        inference_bench(**{**options, "quality_seed": 42})


@pytest.mark.parametrize("configuration", ["default", "original"])
def test_cached_reference_full_setup(reference, monkeypatch, tmp_path, configuration):
    """Drive both configurations through cached runtime preparation without fetching prompts or recomputing quality."""
    options, destination, _, _ = reference
    options = {**load_config(f"run_configs/{configuration}.yaml")["task"]["args"],
               "quality_samples": 2, "quality_cache_dir": options["quality_cache_dir"],
               "scenarios": "A", "seed_pairs": [[999, 777]] if configuration == "original" else [[21, 1337]]}
    module = importlib.import_module("inferencebench.environment")
    remote = tmp_path / "remote"
    remote.mkdir()
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    upstream = tmp_path / "upstream"
    scenario_dir = upstream / "src/eval/tasks" / "scenario_a"
    # Resolve the real upstream scenario directory from task metadata.
    task = inference_bench(**options)
    scenario_dir = upstream / "src/eval/tasks" / task.dataset[0].metadata["directory"]
    scenario_dir.mkdir(parents=True)
    for name in ["scenario.json", "mission.txt", "benchmark.txt"]:
        (scenario_dir / name).write_text("{}")
    monkeypatch.setattr(runtime, "ARTIFACTS", remote)
    monkeypatch.setattr(runtime, "TASK", task_dir)
    monkeypatch.setattr(runtime, "ROOT", upstream)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=Mock(return_value="/snapshots/fixture-revision")))
    quality_runner = Mock(side_effect=AssertionError("quality must not be recomputed"))
    cache_samples = SimpleNamespace(cache_mmlu_pro=Mock(side_effect=AssertionError("questions must not be fetched")))

    def write_requests(path, rows):
        """Persist the requested speed workload using the upstream JSONL contract."""
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    runner = SimpleNamespace(load_scenario_config=lambda path: {},
                             _get_tokenizer=lambda model: "tokenizer",
                             _prepare_requests=Mock(side_effect=AssertionError("long prompts must not be fetched")),
                             _load_requests_jsonl=lambda path, *args: ([json.loads(line) for line in path.read_text().splitlines()], []))
    monkeypatch.setitem(sys.modules, "inference", SimpleNamespace(
        baseline_eval=SimpleNamespace(_run_baseline=lambda *args, **kwargs: {"profiles": {"burst": {"success_count": 1}}}),
        cache_samples=cache_samples, precompute_baseline=SimpleNamespace(_write_requests_jsonl=write_requests),
        quality_gate=SimpleNamespace(get_quality_specs=lambda: ([SimpleNamespace(samples_file=remote / "quality-samples.jsonl")], None, None)),
        runner=runner,
    ))
    from contextlib import nullcontext
    monkeypatch.setattr(runtime, "baseline_server", lambda options: nullcontext())
    monkeypatch.setattr(runtime, "prepare_quality_baseline", quality_runner)
    monkeypatch.setattr(runtime, "install_workspace", lambda options: None)

    async def write_file(path, content):
        """Map real host transfers into the isolated sandbox filesystem."""
        (remote / Path(path).name).write_text(content)

    async def upload(source, target):
        """Copy the validated cached samples and registry through the sandbox upload boundary."""
        shutil.copyfile(source, remote / Path(target).name)

    async def execute(command, **kwargs):
        """Execute the genuine runtime preparation when Inspect invokes the remote adapter."""
        if "prepare" in command:
            runtime.prepare(json.loads((remote / "options.json").read_text()))
        return ExecResult(True, 0, "", "")

    env = SimpleNamespace(resource_id="mock-cache-sandbox", exec=execute, write_file=write_file,
                          upload=upload, read_file=AsyncMock(side_effect=lambda path: (remote / Path(path).name).read_text()),
                          download=AsyncMock())
    monkeypatch.setattr(module, "gpu_environment", lambda: env)
    task.sandbox = None
    task.scorer = None
    [log] = inspect_eval(task, solver=generate(), model="mockllm/cache-test", display="none", log_dir="logs")
    try:
        assert log.status == "success", log.samples[0].error
        observed = json.loads((remote / "quality.json").read_text())
        assert observed["datasets"]["mmlu_pro"][0]["accuracy"] == 0.5
        quality_runner.assert_not_called()
        runner._prepare_requests.assert_not_called()
        expected_requests = options["request_limit"] or SCENARIOS["A"]["config"]["num_requests"]
        for name in ["dev", "heldout"]:
            assert len((remote / f"{name}-requests.jsonl").read_text().splitlines()) == expected_requests
        assert len(log.samples[0].messages) >= 2
        assert log.samples[0].metadata["quality_cache_provenance"]["identity"] == quality_identity(options)
    finally:
        Path(log.location).unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_preparation_lifecycle(reference, monkeypatch, tmp_path):
    """Collect measured reference artifacts and release only the preparation sandbox on success or failure."""
    options, _, source, _ = reference
    module = importlib.import_module("inferencebench.prepare_quality")

    async def download(remote, local):
        """Return controlled completed measurements through the real collection boundary."""
        shutil.copyfile(source / Path(remote).name, local)

    env = SimpleNamespace(resource_id="preparation-only", write_file=AsyncMock(), download=download, read_file=AsyncMock(return_value="prepared"),
                          exec=AsyncMock(return_value=ExecResult(True, 0, "", "")), terminate=AsyncMock())
    provider = SimpleNamespace(task_init=AsyncMock(), sample_init=AsyncMock(return_value={"default": env}),
                               task_cleanup=AsyncMock())
    monkeypatch.setattr(module, "InferenceSandbox", provider)
    for fail in [False, True]:
        folder = tmp_path / str(fail)
        folder.mkdir()
        if fail:
            env.exec.side_effect = RuntimeError("preparation interrupted")
            with pytest.raises(RuntimeError, match="interrupted"):
                await measure_reference(options, folder)
        else:
            await measure_reference(options, folder)
            assert read_reference(folder, options)[2] == 0.5
        assert env.terminate.await_count == int(fail) + 1


def test_quality_only_runtime(reference, monkeypatch, tmp_path):
    """Run the standalone quality adapter without long-prompt sampling or speed inference."""
    options, _, source, _ = reference
    remote = tmp_path / "quality-only"
    remote.mkdir()
    from contextlib import nullcontext
    monkeypatch.setattr(runtime, "ARTIFACTS", remote)
    monkeypatch.setattr(runtime, "baseline_server", lambda options: nullcontext())
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=lambda *args, **kwargs: "/snapshots/fixture-revision"))
    spec = SimpleNamespace(samples_file=remote / "quality-samples.jsonl", seed=248, limit=2)

    def select_samples(path, seed, count):
        """Select the controlled questions at the upstream sampling boundary."""
        assert seed == 248 and count == 2
        shutil.copyfile(source / "quality-samples.jsonl", path)

    def run_dataset(spec, url, model, timeout, concurrency, out):
        """Return the original adapter's per-question measurement shape."""
        path = out / "baseline_generations.jsonl"
        shutil.copyfile(source / "resolved_generations.jsonl", path)
        return 0.5, path, None

    monkeypatch.setitem(sys.modules, "inference", SimpleNamespace(
        cache_samples=SimpleNamespace(cache_mmlu_pro=select_samples),
        quality_gate=SimpleNamespace(get_quality_specs=lambda: ([spec], None, None)),
        precompute_quality_baseline=SimpleNamespace(_run_dataset=run_dataset),
    ))
    def system_info(command, **kwargs):
        """Supply deterministic evaluator, library, and hardware provenance for the runtime adapter."""
        if command[0] == "git":
            return quality_identity(options)["upstream_revision"]
        if command[0] == "python3":
            return json.dumps({"torch": "fixture", "transformers": "fixture"})
        return "mock H100"

    monkeypatch.setattr(runtime.subprocess, "check_output", system_info)
    runtime.quality_reference(options)
    assert read_reference(remote, options)[2] == 0.5
    assert not (remote / "baseline.json").exists()
    assert json.loads((remote / "provenance.json").read_text())["downloaded_model_revision"] == "fixture-revision"


@pytest.mark.parametrize("configuration", ["default", "original"])
def test_configs_share_quality_reference(reference, configuration):
    """Select the same cached quality reference while preserving each configuration's long-prompt settings."""
    options, destination, _, _ = reference
    config = load_config(f"run_configs/{configuration}.yaml")["task"]["args"]
    task = inference_bench(**{**config, "quality_samples": 2, "quality_cache_dir": options["quality_cache_dir"]})
    assert task.dataset[0].metadata["quality_cache"] == str(destination)
    assert task.dataset[0].metadata["quality_seed"] == 248
    assert task.dataset[0].metadata["request_limit"] == config["request_limit"]
    assert Path(task.dataset[0].metadata["request_cache"]).is_file()


def test_local_reference_overrides_bundle(reference, monkeypatch, tmp_path):
    """Allow a newly prepared local reference to replace an unusable bundled copy."""
    options, destination, _, _ = reference
    broken_bundle = tmp_path / "broken-bundle"
    broken_bundle.mkdir()
    (broken_bundle / "manifest.json").write_text("invalid json")
    module = importlib.import_module("inferencebench.quality_cache")
    monkeypatch.setattr(module, "BUNDLED_QUALITY", broken_bundle)
    assert load_quality_cache(options) == destination


def test_failed_rebuild_preserves_cache(reference):
    """Keep the previous usable reference when a replacement fails completeness validation."""
    options, destination, source, _ = reference
    original = (destination / "manifest.json").read_bytes()
    (source / "resolved_generations.jsonl").write_text("")
    with pytest.raises(ValueError, match="every selected question"):
        publish_reference(source, options, destination, Path("logs").resolve(), replace=True)
    assert (destination / "manifest.json").read_bytes() == original
    assert load_quality_cache(options) == destination


@pytest.mark.parametrize("configuration", ["default", "original"])
def test_bundled_reference(configuration, tmp_path):
    """Load the shipped 500-question measurement through each actual configuration without a local override."""
    from inferencebench.quality_cache import BUNDLED_QUALITY

    options = load_config(f"run_configs/{configuration}.yaml")["task"]["args"]
    task = inference_bench(**{**options, "quality_cache_dir": str(tmp_path)})
    assert Path(task.dataset[0].metadata["quality_cache"]) == BUNDLED_QUALITY
    log = read_eval_log(BUNDLED_QUALITY / "reference.eval")
    summary = json.loads((BUNDLED_QUALITY / "quality.json").read_text())
    assert log.status == "success"
    assert log.eval.model == options["base_model"]
    assert len(log.samples) == 500
    assert len({sample.id for sample in log.samples}) == 500
    assert sum(sample.scores["mmlu_pro"].value for sample in log.samples) == summary["datasets"]["mmlu_pro"][0]["correct"]
    assert log.results.scores[0].metrics["accuracy"].value == summary["datasets"]["mmlu_pro"][0]["accuracy"]
