"""Thin remote adapter around the pinned upstream evaluator; this file contains no replacement benchmark logic."""

import argparse
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

ROOT = Path("/opt/inferencebench")
TASK = Path("/home/agent/task")
ARTIFACTS = Path("/tmp/inferencebench")
sys.path.insert(0, str(ROOT / "src/eval"))


def environment(options):
    """Set the original evaluator's explicit environment inputs for this sample."""
    return {
        "HF_HUB_CACHE": str(Path(os.environ["HF_HOME"]) / "hub"),
        "INFERENCE_BENCH_BASE_MODEL": options["base_model"],
        "INFERENCE_BENCH_MAX_MODEL_LEN": str(options["max_model_len"]),
        "INFERENCE_BENCH_ALLOW_HF_DOWNLOAD": "1",
        "INFERENCE_BENCH_DATASET_SEED": str(options["dev_seed"]),
        "INFERENCE_BENCH_QUALITY_SEED": str(options["quality_seed"]),
        "INFERENCE_BENCH_QUALITY_MMLUPRO_N": str(options["quality_samples"]),
        "INFERENCE_BENCH_QUALITY_TAU": str(options["quality_tau"]),
        "INFERENCE_BENCH_QUALITY_CONCURRENCY": str(options["quality_concurrency"]),
        "INFERENCE_BENCH_QUALITY_MMLUPRO_SAMPLES_FILE": str(
            ARTIFACTS / "quality-samples.jsonl"
        ),
        "INFERENCE_BENCH_QUALITY_BASELINE_REGISTRY": str(ARTIFACTS / "quality.json"),
        "INFERENCE_BENCH_SERVER_WAIT_S": str(options["server_wait_seconds"]),
        "INFERENCE_BENCH_REQUEST_TIMEOUT_S": str(options["request_timeout_seconds"]),
    }


def evaluate(options, seed, output, quality):
    """Run the unmodified upstream speed evaluator and optionally its quality gate."""
    from inference import runner

    args = runner.build_parser().parse_args(
        ["--model", options["base_model"], "--seed", str(seed)]
    )
    args.request_limit = options["request_limit"]
    args.requests_file = str(ARTIFACTS / "heldout-requests.jsonl")
    folder = ARTIFACTS / output
    folder.mkdir(parents=True, exist_ok=True)
    metrics = runner.run_speed_eval(TASK, args, folder)

    if quality:
        metrics["quality_check"] = runner.run_quality_eval(
            args.server_url,
            metrics["model_id"],
            args.request_timeout_s,
            folder,
            options["quality_tau"],
        )

    (ARTIFACTS / (output + ".json")).write_text(json.dumps(metrics))
    return metrics


def wait_ready(server, seconds):
    """Stop readiness polling when the launcher exits instead of waiting out a dead server's allowance."""
    deadline = time.monotonic() + seconds
    while server.poll() is None and time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(
                "http://127.0.0.1:8000/v1/models", timeout=1
            ) as response:
                if response.status == 200:
                    return True
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(1)
    return False


def prepare_requests(options, config, name):
    """Load the small prepared prefix or run the original full-corpus sampler explicitly."""
    from inference import runner

    args = (
        config,
        options["request_limit"],
        runner._get_tokenizer(options["base_model"]),
        options["max_model_len"],
    )
    if options["request_cache"] is not None:
        return runner._load_requests_jsonl(ARTIFACTS / f"{name}-requests.jsonl", *args)[0]
    return runner._prepare_requests(*args)[0]


def prepare(options):
    """Cache datasets, measure the Transformers baseline, and install the original empty launcher before timing starts."""
    from huggingface_hub import snapshot_download
    from inference import (
        baseline_eval,
        cache_samples,
        precompute_baseline,
        quality_gate,
        runner,
    )

    # Cache the original datasets and model before benchmarking.
    snapshot = snapshot_download(
        options["base_model"],
        ignore_patterns=["*.pt", "*.bin", "original/*"],
    )
    reference = options.get("quality_cache_provenance")
    if reference is not None and Path(snapshot).name != reference["model_revision"]:
        raise RuntimeError("Model revision differs from the cached MMLU-Pro reference; rebuild the reference before running this model")
    ARTIFACTS.mkdir(exist_ok=True)
    for name in ["scenario.json", "mission.txt", "benchmark.txt"]:
        shutil.copy(ROOT / "src/eval/tasks" / options["directory"] / name, TASK / name)
    if options["request_cache"] is None:
        for seed in {options["dev_seed"], options["eval_seed"]}:
            path = (
                ROOT
                / f"src/eval/inference/baselines/samples/longbench_v2/{seed}_503/samples.jsonl"
            )
            if not path.exists():
                cache_samples.cache_longbench_v2(path, seed, 503)
    specs, _, _ = quality_gate.get_quality_specs()
    spec = specs[0]
    if not spec.samples_file.exists():
        if options.get("quality_cache") is not None:
            raise RuntimeError("Cached MMLU-Pro questions were not transferred to the sandbox")
        cache_samples.cache_mmlu_pro(spec.samples_file, spec.seed, spec.limit)

    with baseline_server(options):
        config = runner.load_scenario_config(TASK)
        config["dataset_seed"] = options["eval_seed"]
        requests = prepare_requests(options, config, "heldout")
        precompute_baseline._write_requests_jsonl(
            ARTIFACTS / "heldout-requests.jsonl", requests
        )
        metrics = baseline_eval._run_baseline(
            "http://127.0.0.1:8000",
            requests,
            options["base_model"],
            ARTIFACTS / "baseline-generations.jsonl",
            TASK,
            options["request_timeout_seconds"],
            concurrency_override=1,
        )
        (ARTIFACTS / "baseline.json").write_text(json.dumps(metrics))
        if any(
            profile["success_count"] == 0
            for profile in metrics["profiles"].values()
        ):
            raise RuntimeError(
                "Transformers baseline returned no successful requests"
            )

        if options.get("quality_cache") is None:
            prepare_quality_baseline(options, spec)

    # Give the agent a separate development request set.
    config["dataset_seed"] = options["dev_seed"]
    dev_requests = prepare_requests(options, config, "dev")
    precompute_baseline._write_requests_jsonl(TASK / "requests.jsonl", dev_requests)
    # Record resolved weights and the exact evaluated inputs without claiming historical data pins.
    provenance = {"downloaded_model_revision": Path(snapshot).name, "input_sha256": {}}
    if options.get("quality_cache_provenance") is not None:
        provenance["quality_cache"] = options["quality_cache_provenance"]
    if options.get("request_cache_provenance") is not None:
        provenance["request_cache"] = options["request_cache_provenance"]
    for path in [
        TASK / "requests.jsonl",
        ARTIFACTS / "heldout-requests.jsonl",
        spec.samples_file,
    ]:
        provenance["input_sha256"][str(path)] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    (ARTIFACTS / "provenance.json").write_text(json.dumps(provenance, indent=2))
    install_workspace(options)


@contextmanager
def baseline_server(options):
    """Run the original float16 Transformers server and release it after reference measurement."""
    command = [
        "python3",
        str(ROOT / "src/eval/inference/servers/transformers_openai_server.py"),
        "--model",
        options["base_model"],
        "--port",
        "8000",
        "--dtype",
        "float16",
        "--max-model-len",
        str(options["max_model_len"]),
    ]
    with (ARTIFACTS / "baseline-server.log").open("w") as log:
        server = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            if not wait_ready(server, options["server_wait_seconds"]):
                raise RuntimeError(
                    "Transformers baseline failed to start: "
                    + (ARTIFACTS / "baseline-server.log").read_text()
                )
            yield
        finally:
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait()


def quality_reference(options):
    """Measure only the MMLU-Pro reference, preserving all requests and model provenance."""
    from huggingface_hub import snapshot_download
    from inference import cache_samples, quality_gate

    started = time.time()
    ARTIFACTS.mkdir(exist_ok=True)
    snapshot = snapshot_download(
        options["base_model"], ignore_patterns=["*.pt", "*.bin", "original/*"]
    )
    specs, _, _ = quality_gate.get_quality_specs()
    spec = specs[0]
    cache_samples.cache_mmlu_pro(spec.samples_file, spec.seed, spec.limit)
    with baseline_server(options):
        prepare_quality_baseline(options, spec)
    shutil.copy(ARTIFACTS / "quality-baseline/resolved_generations.jsonl", ARTIFACTS / "resolved_generations.jsonl")
    provenance = {
        "downloaded_model_revision": Path(snapshot).name,
        "started_at_unix": started,
        "completed_at_unix": time.time(),
        "upstream_revision": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "reference_options": options,
        "server_versions": json.loads(subprocess.check_output([
            "python3", "-c",
            "import importlib.metadata as m, json; print(json.dumps({name: m.version(name) for name in ['torch', 'transformers']}))",
        ], text=True)),
        "input_sha256": {"quality-samples.jsonl": hashlib.sha256(spec.samples_file.read_bytes()).hexdigest()},
        "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"], text=True),
    }
    (ARTIFACTS / "provenance.json").write_text(json.dumps(provenance, indent=2))


def prepare_quality_baseline(options, spec):
    """Require a complete, usable quality reference before starting agent optimization."""
    from inference import precompute_quality_baseline

    out = ARTIFACTS / "quality-baseline"
    out.mkdir(exist_ok=True)
    accuracy, log_path, _ = precompute_quality_baseline._run_dataset(
        spec,
        "http://127.0.0.1:8000",
        options["base_model"],
        options["request_timeout_seconds"],
        options["quality_concurrency"],
        out,
    )
    results = [json.loads(line) for line in log_path.read_text().splitlines()]
    if len(results) != spec.limit or not any(row["success"] for row in results):
        raise RuntimeError("Transformers quality baseline did not complete every request")
    if not all(row["success"] for row in results):
        results = retry_quality_reference(options, spec, results, out)
        if all(row["success"] for row in results):
            accuracy = precompute_quality_baseline._accuracy(results)
    if not all(row["success"] for row in results):
        raise RuntimeError("Transformers quality baseline did not complete every request")
    (out / "resolved_generations.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in results)
    )
    # The upstream relative-quality gate is undefined for a zero reference.
    if not math.isfinite(accuracy) or not 0 < accuracy <= 1:
        raise RuntimeError(f"Transformers quality baseline has unusable accuracy: {accuracy}")

    registry = {
        "datasets": {
            "mmlu_pro": [{"seed": spec.seed, "n": spec.limit, "accuracy": accuracy}]
        }
    }
    (ARTIFACTS / "quality.json").write_text(json.dumps(registry))


def retry_quality_reference(options, spec, results, out):
    """Retry only failed reference requests in isolation, retaining every original attempt."""
    from inference import precompute_quality_baseline

    if options["quality_baseline_max_attempts"] == 1:
        return results
    samples = [json.loads(line) for line in spec.samples_file.read_text().splitlines()]
    by_id = {str(row["sample_id"]): row for row in samples}
    if len(by_id) != len(samples) or len({row["sample_id"] for row in results}) != spec.limit:
        raise RuntimeError("Transformers quality baseline has duplicate sample IDs")
    results = list(results)
    for attempt in range(2, options["quality_baseline_max_attempts"] + 1):
        for index, previous in enumerate(results):
            if previous["success"]:
                continue
            retry_dir = out / f"attempt-{attempt}" / str(index)
            retry_dir.mkdir(parents=True)
            sample_file = retry_dir / "samples.jsonl"
            sample_file.write_text(json.dumps(by_id[previous["sample_id"]]) + "\n")
            retry_spec = replace(spec, samples_file=sample_file, limit=1)
            _, log_path, _ = precompute_quality_baseline._run_dataset(
                retry_spec, "http://127.0.0.1:8000", options["base_model"],
                options["request_timeout_seconds"], 1, retry_dir,
            )
            retried = [json.loads(line) for line in log_path.read_text().splitlines()]
            if len(retried) != 1 or retried[0]["sample_id"] != previous["sample_id"]:
                raise RuntimeError("Transformers quality baseline retry returned mismatched requests")
            results[index] = {**retried[0], "request_index": previous["request_index"]}
    return results


def install_workspace(options):
    """Install the original task launchers and a development evaluator with overridable environment defaults."""
    context = ROOT / "src/eval/tasks/_shared/task_context"
    for name in ["start_server.sh", "test_server.sh"]:
        shutil.copy(context / name, TASK / name)
        (TASK / name).chmod(0o755)

    # Development tests can override these defaults through the environment.
    env = environment(options)
    if options["request_cache"] is not None:
        env["INFERENCE_BENCH_REQUESTS_FILE"] = str(TASK / "requests.jsonl")
    wrapper = "#!/opt/evaluator/bin/python\nimport os, sys\nfrom pathlib import Path\n"
    wrapper += f"for key, value in {env!r}.items(): os.environ.setdefault(key, value)\nsys.path.insert(0, {str(ROOT / 'src/eval')!r})\n"
    wrapper += "from inference.runner import build_parser, run_evaluation\nrun_evaluation(Path(__file__).parent, build_parser().parse_args())\n"
    (TASK / "evaluate.py").write_text(wrapper)
    (TASK / "evaluate.py").chmod(0o755)


def final(options):
    """Relaunch the submitted server under upstream supervision and produce fresh held-out measurements."""
    for name in ["scenario.json", "mission.txt", "benchmark.txt"]:
        shutil.copy(ROOT / "src/eval/tasks" / options["directory"] / name, TASK / name)

    # Score a fresh launch against the restored held-out requests.
    with (ARTIFACTS / "final-server.log").open("w") as log:
        server = subprocess.Popen(
            [
                "bash",
                "/opt/inference_eval/bin/launch_supervised_server.sh",
                str(TASK / "start_server.sh"),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            if not wait_ready(server, options["server_wait_seconds"]):
                (ARTIFACTS / "final.json").write_text(
                    json.dumps(
                        {
                            "invalid_submission": "Server did not become ready after a clean restart"
                        }
                    )
                )
                return
            try:
                metrics = evaluate(options, options["eval_seed"], "final", True)
            except TimeoutError as error:
                # The upstream readiness timeout names the server that disappeared.
                if server.poll() is None or not str(error).startswith(
                    "Timed out waiting for server at "
                ):
                    raise
                metrics = {
                    "evaluator_error": repr(error),
                    "launcher_returncode": server.returncode,
                }
            if server.poll() is not None:
                metrics["invalid_submission"] = (
                    "Canonical launcher exited during final evaluation"
                )
                (ARTIFACTS / "final.json").write_text(json.dumps(metrics))
        finally:
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait()


def main():
    """Dispatch a trusted preparation or final-scoring operation from explicit JSON options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=["prepare", "quality-reference", "final"])
    parser.add_argument("options")
    args = parser.parse_args()
    options = json.loads(Path(args.options).read_text())
    os.environ.update(environment(options))
    {"prepare": prepare, "quality-reference": quality_reference, "final": final}[args.operation](options)


if __name__ == "__main__":
    main()
