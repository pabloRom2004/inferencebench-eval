"""Thin remote wrapper that drives the pinned upstream evaluator's own entrypoints; it contains no replacement benchmark logic."""

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from contextlib import contextmanager, nullcontext
from pathlib import Path

# The patched upstream copy, and the read-only bundle upstream's harness binds at /opt/inference_eval.
ROOT = Path("/opt/inferencebench")
BUNDLE = Path("/opt/inference_eval")
INFERENCE = ROOT / "src/eval/inference"
TASK = Path("/home/agent/task")
ARTIFACTS = Path("/tmp/inferencebench")
TRUSTED = ARTIFACTS / "trusted"
REFERENCE_BIN = Path("/opt/reference/bin")
# Login shells (Inspect's bash tool) source this, overriding the image's static defaults.
PROFILE = Path("/etc/profile.d/inferencebench.sh")
SERVER_URL = "http://127.0.0.1:8000"
# Upstream's baseline orchestrator gives the sequential Transformers server a longer request timeout.
TORCH_BASELINE_TIMEOUT_S = 900
# Files upstream's harness copies into the read-only evaluator bundle.
BUNDLE_FILES = ["__init__.py", "runner.py", "quality_gate.py", "cache_samples.py"]


def model_safe(model: str) -> str:
    """Name model folders the way upstream's precompute scripts do."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model)


def num_hours(options) -> str | None:
    """Render the wall-clock budget as upstream's NUM_HOURS argument, or None without a deadline."""
    seconds = options["agent_seconds"]
    return None if seconds is None else f"{seconds / 3600:g}"


def environment(options):
    """Set the original harness's agent-container environment for this sample."""
    env = {
        "HF_HUB_CACHE": str(Path(os.environ["HF_HOME"]) / "hub"),
        "INFERENCE_BENCH_BASE_MODEL": options["base_model"],
        "INFERENCE_BENCH_MAX_MODEL_LEN": str(options["max_model_len"]),
        "INFERENCE_BENCH_INPUT_TOKEN_MARGIN": "16",
        "INFERENCE_BENCH_ALLOW_HF_DOWNLOAD": "1",
        "INFERENCE_BENCH_SCENARIO": options["directory"],
        "INFERENCE_BENCH_DATASET_SEED": str(options["dev_seed"]),
        "INFERENCE_BENCH_EVAL_SEED": str(options["eval_seed"]),
        "INFERENCE_BENCH_QUALITY_TAU": str(options["quality_tau"]),
        "INFERENCE_BENCH_QUALITY_SEED": str(options["quality_seed"]),
        "INFERENCE_BENCH_QUALITY_MMLUPRO_N": str(options["quality_samples"]),
        "INFERENCE_BENCH_QUALITY_CONCURRENCY": str(options["quality_concurrency"]),
        "INFERENCE_BENCH_QUALITY_BASELINE_REGISTRY": "",
        "INFERENCE_BENCH_QUALITY_BASELINE_BACKEND": reference_backend(options),
        "INFERENCE_BENCH_SERVER_URL": SERVER_URL,
        "INFERENCE_BENCH_SERVER_HOST": "127.0.0.1",
        "INFERENCE_BENCH_SERVER_PORT": "8000",
        "HOST": "127.0.0.1",
        "PORT": "8000",
        "INFERENCE_BENCH_METRICS_PATH": str(TASK / "metrics_preview.json"),
        "INFERENCE_BENCH_SERVER_WAIT_S": str(options["server_wait_seconds"]),
        "INFERENCE_BENCH_REQUEST_TIMEOUT_S": str(options["request_timeout_seconds"]),
        "INFERENCE_BENCH_STARTING_POINT": "default",
    }
    if num_hours(options) is not None:
        env["NUM_HOURS"] = num_hours(options)
    env.update(evaluator_env(options, options["dev_seed"]))
    return env


def evaluator_env(options, seed) -> dict[str, str]:
    """Configure the patched evaluator's arrival seed, Scenario A output cap, and output token counting when enabled."""
    env = {"INFERENCE_BENCH_ARRIVAL_SEED": str(seed)} if options["seeded_arrivals"] else {}
    if options["retokenize_outputs"]:
        env["INFERENCE_BENCH_RETOKENIZE_OUTPUTS"] = "1"
    if options["scenario"] == "A" and options["scenario_a_output_tokens"] is not None:
        env["INFERENCE_BENCH_OUTPUT_TOKEN_CAP"] = str(options["scenario_a_output_tokens"])
    return env


def reference_backend(options) -> str:
    """Name the quality-reference backend the way upstream's registry files do."""
    return "torch" if options["quality_reference_backend"] == "transformers" else "vllm"


def registry_name(options) -> str:
    """The quality registry filename upstream's gate resolves for the configured backend."""
    suffix = "" if reference_backend(options) == "vllm" else "_" + reference_backend(options)
    return f"{model_safe(options['base_model'])}{suffix}.json"


def speed_folder(options) -> Path:
    """Where upstream's precompute writes, and its final evaluation reads, the held-out speed requests."""
    return INFERENCE / "baselines/speed/torch" / options["directory"] / model_safe(options["base_model"])


def run_upstream(command, log_name, env=None, timeout=None, check=True):
    """Run one upstream entrypoint from the repository root, keeping its output for the run's artifacts."""
    ARTIFACTS.mkdir(exist_ok=True)
    with (ARTIFACTS / log_name).open("a") as log:
        log.write("$ " + shlex.join(command) + "\n")
        log.flush()
        result = subprocess.run(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, **(env or {})}, timeout=timeout,
        )
    if check and result.returncode != 0:
        tail = (ARTIFACTS / log_name).read_text().splitlines()[-40:]
        raise RuntimeError(f"Upstream command failed ({result.returncode}): {shlex.join(command)}\n" + "\n".join(tail))
    return result.returncode


def install(options):
    """Unpack the untouched upstream copy, apply the port's patches, and start the read-only evaluator bundle."""
    staging = ARTIFACTS / "install"
    shutil.rmtree(staging, ignore_errors=True)
    with tarfile.open(ARTIFACTS / "upstream.tar") as archive:
        archive.extractall(staging, filter="data")
    shutil.rmtree(ROOT, ignore_errors=True)
    shutil.move(str(staging / "upstream"), ROOT)
    for patch in sorted((staging / "patches").glob("*.patch")) if (staging / "patches").is_dir() else []:
        subprocess.run(["git", "apply", str(patch)], cwd=ROOT, check=True)
    shutil.rmtree(BUNDLE, ignore_errors=True)
    (BUNDLE / "bin").mkdir(parents=True)
    shutil.copy(INFERENCE / "bin/launch_supervised_server.sh", BUNDLE / "bin")
    (BUNDLE / "bin/launch_supervised_server.sh").chmod(0o755)


def cache_samples(options):
    """Pre-cache the seeded MMLU-Pro questions and LongBench-v2 pools with upstream's own sampler command."""
    quality = ["--mmlupro-n", str(options["quality_samples"])]
    run_upstream([sys.executable, "-m", "src.eval.inference.cache_samples", "--seed", str(options["quality_seed"]), *quality], "cache-samples.log")
    for seed in sorted({options["dev_seed"], options["eval_seed"]}):
        run_upstream([sys.executable, "-m", "src.eval.inference.cache_samples", "--seed", str(seed), *quality, "--longbench-n", "503"], "cache-samples.log")


def speed_baseline(options):
    """Measure the PyTorch speed baseline with upstream's precompute command, or install the shared measurement."""
    folder = speed_folder(options)
    folder.mkdir(parents=True, exist_ok=True)
    cached = ARTIFACTS / "cached" / "speed"
    if options.get("cached_speed_baseline"):
        for name in ["requests.jsonl", "baseline_metrics.json"]:
            if not (cached / name).is_file():
                raise RuntimeError("Shared speed baseline files were not transferred to the sandbox")
            shutil.copy(cached / name, folder / name)
        return folder
    command = [
        sys.executable, "-m", "src.eval.inference.precompute_baseline",
        "--scenario-id", options["directory"],
        "--base-model", options["base_model"],
        "--server-url", SERVER_URL,
        "--out-root", str(folder.parents[1]),
        "--registry", str(folder.parents[1] / f"{model_safe(options['base_model'])}.json"),
        "--seed", str(options["eval_seed"]),
        "--request-timeout-s", str(TORCH_BASELINE_TIMEOUT_S),
        "--concurrency-override", "1",
    ]
    run_upstream(command, "speed-baseline.log", env=evaluator_env(options, options["eval_seed"]))
    return folder


def quality_reference(options):
    """Measure the MMLU-Pro reference with upstream's precompute command, then complete any failed requests in isolation."""
    backend = reference_backend(options)
    registry = INFERENCE / "baselines/quality" / registry_name(options)
    out_root = INFERENCE / "baselines/quality" / backend / model_safe(options["base_model"])
    concurrency = "1" if backend == "torch" else str(options["quality_concurrency"])
    command = [
        sys.executable, "-m", "src.eval.inference.precompute_quality_baseline",
        "--server-url", SERVER_URL,
        "--base-model", options["base_model"],
        "--backend", backend,
        "--registry", str(registry),
        "--out-root", str(out_root),
        "--seed", str(options["quality_seed"]),
        "--mmlupro-n", str(options["quality_samples"]),
        "--request-timeout-s", str(options["request_timeout_seconds"]),
        "--concurrency", concurrency,
    ]
    run_upstream(command, "quality-reference.log")
    generations = out_root / "mmlu_pro" / f"{options['quality_seed']}_{options['quality_samples']}" / "baseline_generations.jsonl"
    rows = [json.loads(line) for line in generations.read_text().splitlines()]
    retried = 0
    if options["quality_baseline_max_attempts"] > 1:
        rows, retried = retry_quality_reference(options, command, rows)
        if len(rows) != options["quality_samples"] or not all(row["success"] for row in rows):
            raise RuntimeError("Transformers quality baseline did not complete every request")
        (generations.with_name("resolved_generations.jsonl")).write_text("".join(json.dumps(row) + "\n" for row in rows))
        if retried:
            sys.path.insert(0, str(ROOT / "src/eval"))
            from inference import precompute_quality_baseline

            data = json.loads(registry.read_text())
            for entry in data["datasets"]["mmlu_pro"]:
                if int(entry["seed"]) == options["quality_seed"] and int(entry["n"]) == options["quality_samples"]:
                    entry["accuracy"] = precompute_quality_baseline._accuracy(rows)
                    entry["note"] = f"{retried} failed request(s) retried in isolation by the Inspect port"
            registry.write_text(json.dumps(data, indent=2))
    return {"registry": registry, "generations": generations, "concurrency": int(concurrency), "retried_requests": retried, "complete": all(row["success"] for row in rows)}


def retry_quality_reference(options, command, rows):
    """Re-run upstream's precompute for each failed question alone, keeping every completed answer as it was."""
    sys.path.insert(0, str(ROOT / "src/eval"))
    from inference import quality_gate

    spec = quality_gate.get_quality_specs()[0][0]
    samples = [json.loads(line) for line in Path(spec.samples_file).read_text().splitlines()]
    by_id = {str(row["sample_id"]): row for row in samples}
    if len(by_id) != len(samples) or len({row["sample_id"] for row in rows}) != len(rows):
        raise RuntimeError("Transformers quality baseline has duplicate sample IDs")
    rows, retried = list(rows), 0
    for attempt in range(2, options["quality_baseline_max_attempts"] + 1):
        for index, previous in enumerate(rows):
            if previous["success"]:
                continue
            retry = ARTIFACTS / "quality-retries" / f"attempt-{attempt}" / str(index)
            retry.mkdir(parents=True)
            (retry / "samples.jsonl").write_text(json.dumps(by_id[previous["sample_id"]]) + "\n")
            retry_command = list(command)
            for flag, value in [("--registry", retry / "registry.json"), ("--out-root", retry), ("--mmlupro-n", 1), ("--concurrency", 1)]:
                retry_command[retry_command.index(flag) + 1] = str(value)
            run_upstream(retry_command, "quality-reference.log", env={"INFERENCE_BENCH_QUALITY_MMLUPRO_SAMPLES_FILE": str(retry / "samples.jsonl")})
            [result] = [json.loads(line) for line in next(retry.glob("mmlu_pro/*/baseline_generations.jsonl")).read_text().splitlines()]
            if result["sample_id"] != previous["sample_id"]:
                raise RuntimeError("Transformers quality baseline retry returned mismatched requests")
            rows[index] = {**result, "request_index": previous["request_index"]}
            retried += 1
    return rows, retried


def assemble_bundle():
    """Populate the read-only evaluator bundle the way upstream's harness does, with this run's caches and registries."""
    BUNDLE.mkdir(parents=True, exist_ok=True)
    for name in BUNDLE_FILES:
        shutil.copy(INFERENCE / name, BUNDLE / name)
    for folder in ["quality", "samples"]:
        shutil.rmtree(BUNDLE / "baselines" / folder, ignore_errors=True)
        shutil.copytree(INFERENCE / "baselines" / folder, BUNDLE / "baselines" / folder)


def prepare(options):
    """Cache inputs, measure or install the speed baseline, measure the quality reference, and set up the original workspace before timing starts."""
    from huggingface_hub import snapshot_download

    ARTIFACTS.mkdir(exist_ok=True)
    snapshot = snapshot_download(options["base_model"], ignore_patterns=["*.pt", "*.bin", "original/*"])
    speed_cached = bool(options.get("cached_speed_baseline"))
    transformers_reference = options["quality_reference_backend"] == "transformers"
    cache_samples(options)

    shutil.rmtree(TRUSTED, ignore_errors=True)
    (TRUSTED / "speed").mkdir(parents=True)
    (TRUSTED / "quality").mkdir()
    # The Transformers server serves the speed baseline and, in the original configuration, the reference.
    with baseline_server(options) if not speed_cached or transformers_reference else nullcontext():
        speed = speed_baseline(options)
        for name in ["requests.jsonl", "baseline_metrics.json", "baseline_generations.jsonl"]:
            if (speed / name).is_file():
                shutil.copy(speed / name, TRUSTED / "speed" / name)
        if transformers_reference:
            reference = quality_reference(options)
    vllm_version = None
    if not transformers_reference:
        with reference_server(options) as vllm_version:
            reference = quality_reference(options)

    sys.path.insert(0, str(ROOT / "src/eval"))
    from inference import quality_gate

    samples = Path(quality_gate.get_quality_specs()[0][0].samples_file)
    lock = json.loads((ARTIFACTS / "install/upstream.lock").read_text())
    provenance = {
        "upstream": {**lock, "patches": [
            {"name": patch.name, "sha256": hashlib.sha256(patch.read_bytes()).hexdigest()}
            for patch in sorted((ARTIFACTS / "install/patches").glob("*.patch"))
        ]},
        "downloaded_model_revision": Path(snapshot).name,
        "input_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in [speed / "requests.jsonl", samples]},
        "quality_reference": {
            "backend": options["quality_reference_backend"],
            "vllm_version": vllm_version,
            "concurrency": reference["concurrency"],
            "retried_requests": reference["retried_requests"],
            "complete": reference["complete"],
        },
        "speed_baseline": "cached" if speed_cached else "measured",
    }

    # Everything final scoring restores from the controller after the restart.
    shutil.copy(reference["registry"], TRUSTED / "quality" / reference["registry"].name)
    for name in ["baseline_generations.jsonl", "resolved_generations.jsonl"]:
        if reference["generations"].with_name(name).is_file():
            shutil.copy(reference["generations"].with_name(name), TRUSTED / "quality" / name)
    shutil.copy(samples, TRUSTED / "quality" / "samples.jsonl")
    (TRUSTED / "provenance.json").write_text(json.dumps(provenance, indent=2))
    (TRUSTED / "environment.json").write_text(json.dumps(environment(options), indent=2))

    assemble_bundle()
    install_workspace(options)


def evaluate_stub() -> str:
    """The evaluate.py upstream's harness installs in every agent workspace, taken from its runner script's heredoc."""
    script = (ROOT / "src/run_task.sh").read_text()
    marker = "task/evaluate.py\" <<'PY'\n"
    start = script.index(marker) + len(marker)
    return script[start:script.index("\nPY\n", start)] + "\n"


def install_workspace(options):
    """Install the original task files, launch scaffold, evaluator stub, and agent environment like upstream's harness."""
    scenario = ROOT / "src/eval/tasks" / options["directory"]
    for name in ["evaluate.py", "mission.txt", "benchmark.txt", "scenario.json"]:
        if (scenario / name).is_file():
            shutil.copy(scenario / name, TASK / name)
    for context in [ROOT / "src/eval/tasks/_shared/task_context", scenario / "task_context"]:
        if context.is_dir():
            shutil.copytree(context, TASK, dirs_exist_ok=True)
    (TASK / "evaluate.py").write_text(evaluate_stub())
    for path in TASK.glob("*.sh"):
        path.chmod(0o755)
    PROFILE.parent.mkdir(parents=True, exist_ok=True)
    PROFILE.write_text("".join(f"export {key}={shlex.quote(value)}\n" for key, value in environment(options).items()))


@contextmanager
def serve(command, log_name, description, options):
    """Run one local server on port 8000 and release the GPU when its measurement ends."""
    with (ARTIFACTS / log_name).open("w") as log:
        server = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            if not wait_ready(server, options["server_wait_seconds"]):
                raise RuntimeError(f"{description} failed to start: " + (ARTIFACTS / log_name).read_text())
            yield
        finally:
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait()


@contextmanager
def baseline_server(options):
    """Run the original Transformers server with upstream's baseline launch command."""
    command = [
        "python3", "-u", "-m", "src.eval.inference.servers.transformers_openai_server",
        "--host", "127.0.0.1", "--port", "8000",
        "--model", options["base_model"],
        "--max-model-len", str(options["max_model_len"]),
        "--dtype", options["baseline_dtype"],
    ]
    with serve(command, "baseline-server.log", "Transformers baseline", options):
        yield


@contextmanager
def reference_server(options):
    """Serve the same checkpoint at the baseline precision through the pinned vLLM environment for a faster quality reference."""
    version = subprocess.check_output([str(REFERENCE_BIN / "python"), "-c", "import vllm; print(vllm.__version__)"], text=True).strip()
    command = [
        str(REFERENCE_BIN / "vllm"), "serve", options["base_model"],
        "--host", "127.0.0.1", "--port", "8000",
        "--dtype", options["baseline_dtype"],
        "--max-model-len", str(options["max_model_len"]),
    ]
    with serve(command, "reference-server.log", "vLLM reference", options):
        yield version


def wait_ready(server, seconds):
    """Stop readiness polling when the launcher exits instead of waiting out a dead server's allowance."""
    deadline = time.monotonic() + seconds
    while server.poll() is None and time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{SERVER_URL}/v1/models", timeout=1) as response:
                if response.status == 200:
                    return True
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(1)
    return False


def restore_trusted(options):
    """Put the controller's copies of the measured inputs back where upstream's final evaluation reads them."""
    speed = speed_folder(options)
    speed.mkdir(parents=True, exist_ok=True)
    for path in (TRUSTED / "speed").iterdir():
        shutil.copy(path, speed / path.name)
    (INFERENCE / "baselines/quality").mkdir(parents=True, exist_ok=True)
    shutil.copy(TRUSTED / "quality" / registry_name(options), INFERENCE / "baselines/quality" / registry_name(options))
    samples = INFERENCE / "baselines/samples/mmlu_pro" / f"{options['quality_seed']}_{options['quality_samples']}" / "samples.jsonl"
    samples.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(TRUSTED / "quality/samples.jsonl", samples)


def evaluate(options):
    """Run upstream's final evaluation command with its retry schedule and return the metrics it wrote."""
    output = ARTIFACTS / "final"
    shutil.rmtree(output, ignore_errors=True)
    output.mkdir(parents=True)
    metrics = output / "metrics.json"
    command = [
        sys.executable, str(ROOT / "src/eval/tasks" / options["directory"] / "evaluate.py"),
        "--server-url", SERVER_URL,
        "--json-output-file", str(metrics),
        "--requests-file", str(speed_folder(options) / "requests.jsonl"),
        "--quality-tau", str(options["quality_tau"]),
    ]
    env = {"INFERENCE_BENCH_DATASET_SEED": str(options["eval_seed"]), **evaluator_env(options, options["eval_seed"])}
    # Upstream: three attempts at the configured timeout, then two at 150 seconds, each capped at an hour.
    for extra in [[], [], [], ["--request-timeout-s", "150"], ["--request-timeout-s", "150"]]:
        if metrics.is_file():
            break
        try:
            run_upstream(command + extra, "final-evaluator.log", env=env, timeout=3600, check=False)
        except subprocess.TimeoutExpired:
            pass
    return json.loads(metrics.read_text()) if metrics.is_file() else None


def final(options):
    """Relaunch the submitted server under upstream supervision and produce fresh held-out measurements."""
    restore_trusted(options)
    for name in ["scenario.json", "mission.txt", "benchmark.txt"]:
        shutil.copy(ROOT / "src/eval/tasks" / options["directory"] / name, TASK / name)

    with (ARTIFACTS / "final-server.log").open("w") as log:
        server = subprocess.Popen(
            ["bash", str(BUNDLE / "bin/launch_supervised_server.sh"), str(TASK / "start_server.sh")],
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        try:
            if not wait_ready(server, options["server_wait_seconds"]):
                (ARTIFACTS / "final.json").write_text(json.dumps({"invalid_submission": "Server did not become ready after a clean restart"}))
                return
            metrics = evaluate(options)
            if metrics is None:
                metrics = {"evaluator_error": "no metrics were produced by any evaluation attempt", "launcher_returncode": server.poll()}
            elif metrics.get("error") and not metrics.get("profiles"):
                # Upstream's evaluator records a failed run this way; only a vanished server is the submission's fault.
                if server.poll() is None and "Timed out waiting for server" not in metrics["error"]:
                    raise RuntimeError(f"Final evaluation failed: {metrics['error']}")
                metrics = {"evaluator_error": metrics["error"], "launcher_returncode": server.poll()}
            if server.poll() is not None:
                metrics["invalid_submission"] = "Canonical launcher exited during final evaluation"
            elif "evaluator_error" in metrics:
                metrics["invalid_submission"] = metrics["evaluator_error"]
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
    """Dispatch a trusted installation, preparation, or final-scoring operation from explicit JSON options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=["install", "prepare", "final"])
    parser.add_argument("options")
    args = parser.parse_args()
    options = json.loads(Path(args.options).read_text())
    if args.operation != "install":
        os.environ.update(environment(options))
    {"install": install, "prepare": prepare, "final": final}[args.operation](options)


if __name__ == "__main__":
    main()
