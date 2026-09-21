import hashlib
import io
import json
import os
import re
import shutil
import tarfile
import time
from pathlib import Path
from typing import Any

import anyio
from fsspec.core import url_to_fs
from inspect_ai.hooks import Hooks, hooks
from inspect_ai.log import transcript
from inspect_ai.model import ModelInfo, get_model, get_model_info, set_model_info
from inspect_ai.solver import Solver, solver
from inspect_ai.util import sandbox, store

from inferencebench.dataset import num_hours_text
from inferencebench.modal_sandbox import InferenceSandbox
from inferencebench.prompts import ASSETS
from inferencebench.runpod_sandbox import RunPodSandbox
from inferencebench.vendored import patch_digest, upstream_archive, upstream_lock

REMOTE = "/tmp/inferencebench"
EVALUATOR = "/opt/evaluator/bin/python"
UPSTREAM_ROOT = "/opt/inferencebench"
# Measurements taken once per workload and GPU model and shared by later samples, like upstream's precomputed registries.
BASELINE_CACHE = Path("run-artifacts") / "baselines"
SPEED_BASELINE_FILES = ("requests.jsonl", "baseline_metrics.json")
QUALITY_REFERENCE_FILES = ("samples.jsonl", "baseline_generations.jsonl")


def gpu_model(inventory: str) -> str:
    """Read the GPU model from the nvidia-smi CSV inventory recorded for the sample."""
    lines = [line for line in inventory.splitlines() if line.strip()]
    return lines[1].split(",")[0].strip() if len(lines) > 1 else "unknown"


def upstream_identity() -> dict[str, str]:
    """Name the upstream code a measurement was taken with: the pinned commit and the digest of the applied patches."""
    return {"upstream_commit": upstream_lock()["commit"], "patches": patch_digest()}


def speed_baseline_identity(options: dict[str, Any], gpu: str) -> dict[str, Any]:
    """Identify a reusable Transformers speed baseline by workload, precision, upstream code, and GPU model."""
    return {
        "format_version": 2,
        **upstream_identity(),
        "gpu": gpu,
        **{key: options[key] for key in (
            "base_model", "scenario", "eval_seed", "request_limit", "max_model_len", "baseline_dtype",
        )},
    }


def quality_reference_identity(options: dict[str, Any], gpu: str) -> dict[str, Any]:
    """Identify a reusable MMLU-Pro reference by model, backend, question selection, precision, upstream code, and GPU model; the configuration name, agent, and workload play no part."""
    backend = options["quality_reference_backend"]
    return {
        "format_version": 1,
        "kind": "quality_reference",
        **upstream_identity(),
        "gpu": gpu,
        **{key: options[key] for key in ("base_model", "quality_seed", "quality_samples", "baseline_dtype", "max_model_len")},
        "backend": backend,
        # Upstream's precompute measures the Transformers reference sequentially whatever concurrency is configured.
        "concurrency": 1 if backend == "transformers" else options["quality_concurrency"],
    }


def measurement_folder(identity: dict[str, Any], prefix: str) -> Path:
    """Place each shared measurement in a folder named by its workload and a digest of the full identity."""
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    model = re.sub(r"[^A-Za-z0-9_.-]+", "_", identity["base_model"])
    return BASELINE_CACHE / f"{prefix}-{model}-{digest}"


def speed_baseline_folder(identity: dict[str, Any]) -> Path:
    """The shared folder for one scenario, evaluation seed, and identity."""
    return measurement_folder(identity, f"{identity['scenario']}-seed{identity['eval_seed']}")


def quality_reference_folder(identity: dict[str, Any]) -> Path:
    """The shared folder for one reference backend, question seed and count, and identity."""
    return measurement_folder(identity, f"quality-{identity['backend']}-seed{identity['quality_seed']}-n{identity['quality_samples']}")


def registry_file(folder: Path) -> Path | None:
    """The upstream quality registry stored beside a shared reference, whatever backend suffix it carries."""
    registries = [path for path in folder.glob("*.json") if path.name != "manifest.json"]
    return registries[0] if len(registries) == 1 else None


def cached_measurement(folder: Path, identity: dict[str, Any], required: tuple[str, ...]) -> Path | None:
    """Return the stored folder only when its manifest matches this identity and its files exist."""
    manifest = folder / "manifest.json"
    if not manifest.is_file():
        return None
    try:
        recorded = json.loads(manifest.read_text()).get("identity")
    except ValueError:
        return None
    if recorded != identity or not all((folder / name).is_file() for name in required):
        return None
    return folder


def cached_speed_baseline(identity: dict[str, Any]) -> Path | None:
    """The stored speed baseline for this identity, if complete."""
    return cached_measurement(speed_baseline_folder(identity), identity, SPEED_BASELINE_FILES)


def cached_quality_reference(identity: dict[str, Any]) -> Path | None:
    """The stored quality reference for this identity, if complete including its upstream registry."""
    folder = cached_measurement(quality_reference_folder(identity), identity, QUALITY_REFERENCE_FILES)
    return folder if folder is not None and registry_file(folder) is not None else None


def store_measurement(folder: Path, measured: Path, identity: dict[str, Any], provenance: dict[str, Any]) -> Path:
    """Publish a freshly measured folder for later samples, replacing any previous copy atomically."""
    staging = folder.with_name(folder.name + ".staging")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    for path in measured.iterdir():
        if path.is_file():
            shutil.copyfile(path, staging / path.name)
    (staging / "manifest.json").write_text(json.dumps(
        {"identity": identity, "measured_at": time.time(), **provenance}, indent=2,
    ))
    shutil.rmtree(folder, ignore_errors=True)
    staging.rename(folder)
    return folder


def archive_bytes(folder: Path, arcname: str) -> bytes:
    """Pack one local folder for upload into a sandbox."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        archive.add(folder, arcname=arcname)
    return buffer.getvalue()


@hooks(name="inferencebench_artifacts", description="Retain InferenceBench submissions and measurements on Hawk")
class HawkArtifacts(Hooks):
    """Use Hawk's per-sample artifact tree without changing local log placement."""

    def __init__(self):
        self.destinations = {}

    def enabled(self):
        return bool(os.environ.get("HAWK_JOB_ID"))

    async def on_eval_set_start(self, data):
        if data.log_dir.startswith("s3://"):
            self.destinations[data.eval_set_id] = data.log_dir.rstrip("/")

    async def on_sample_end(self, data):
        destination = self.destinations.get(data.eval_set_id)
        folder = data.sample.store.get("artifacts")
        if destination and folder:
            fs, path = url_to_fs(f"{destination}/artifacts/{data.sample_id}")
            await anyio.to_thread.run_sync(lambda: fs.put(folder + "/", path, recursive=True))

    async def on_eval_set_end(self, data):
        self.destinations.pop(data.eval_set_id, None)


def gpu_environment() -> InferenceSandbox | RunPodSandbox:
    """Select the active supported GPU sandbox without changing the agent or evaluator."""
    env = sandbox()
    try:
        return env.as_type(InferenceSandbox)
    except TypeError:
        return env.as_type(RunPodSandbox)


async def checked_exec(env, command: list[str], timeout: int) -> str:
    """Execute harness infrastructure and raise on failure instead of assigning the subject a score."""
    result = await env.exec(command, timeout=timeout, timeout_retry=False)
    if not result.success:
        raise RuntimeError(
            f"Harness command failed ({result.returncode}): {result.stdout}\n{result.stderr}"
        )
    return result.stdout


async def install_upstream(env, options: dict[str, Any]) -> None:
    """Install the untouched upstream copy and the port's patches from this package into the sandbox."""
    await checked_exec(env, ["mkdir", "-p", REMOTE], 30)
    await env.write_file(f"{REMOTE}/runtime.py", (ASSETS / "scripts" / "runtime.py").read_text())
    await env.write_file(f"{REMOTE}/options.json", json.dumps(options))
    await env.write_file(f"{REMOTE}/upstream.tar", upstream_archive())
    await checked_exec(env, [EVALUATOR, f"{REMOTE}/runtime.py", "install", f"{REMOTE}/options.json"], 600)


@solver
def prepare_environment() -> Solver:
    """Prepare the original workload independently of the selected subject solver."""

    async def solve(state, generate):
        """Measure the baseline before starting the optimization clock and retain trusted inputs locally."""
        if state.metadata["context_length"] is not None:
            model = get_model()
            info = get_model_info(model) or ModelInfo()
            set_model_info(str(model), ModelInfo(**{
                **info.model_dump(), "context_length": state.metadata["context_length"],
            }))
        env = gpu_environment()
        folder = (
            Path("run-artifacts")
            / "inferencebench"
            / f"{state.sample_id}-epoch-{state.epoch}-{time.time_ns()}"
        )
        folder.mkdir(parents=True)
        store().set("artifacts", str(folder.resolve()))

        # Record the GPU allocated to this sample.
        state.metadata["agent_sandbox_id"] = env.resource_id
        if isinstance(env, InferenceSandbox):
            state.metadata["modal_sandbox_id"] = env.resource_id
        (folder / "agent-sandbox.json").write_text(
            json.dumps(
                {
                    "sandbox_id": env.resource_id,
                    "provider": state.metadata["gpu_provider"],
                }
            )
        )
        inventory = await checked_exec(
            env,
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv",
            ],
            30,
        )
        (folder / "gpu.txt").write_text(inventory)

        # Reuse measurements already taken for this workload and model on this GPU model.
        gpu = gpu_model(inventory)
        identity = speed_baseline_identity(state.metadata, gpu)
        cached = cached_speed_baseline(identity)
        quality_identity = quality_reference_identity(state.metadata, gpu)
        cached_quality = cached_quality_reference(quality_identity)
        state.metadata["cached_speed_baseline"] = cached is not None
        state.metadata["cached_quality_reference"] = cached_quality is not None
        await install_upstream(env, state.metadata)
        shared = []
        if cached is not None:
            shared.append(("speed", [cached / name for name in SPEED_BASELINE_FILES]))
        if cached_quality is not None:
            shared.append(("quality", [*(cached_quality / name for name in QUALITY_REFERENCE_FILES), registry_file(cached_quality), cached_quality / "manifest.json"]))
        for kind, paths in shared:
            await checked_exec(env, ["mkdir", "-p", f"{REMOTE}/cached/{kind}"], 30)
            for path in paths:
                await env.upload(str(path), f"{REMOTE}/cached/{kind}/{path.name}")

        # Build reference measurements before the agent clock starts.
        try:
            output = await checked_exec(
                env,
                [EVALUATOR, f"{REMOTE}/runtime.py", "prepare", f"{REMOTE}/options.json"],
                43200,
            )
        finally:
            # Preserve measurements and logs even when preparation aborts before optimization.
            with anyio.CancelScope(shield=True):
                try:
                    archive = await env.exec(
                        ["bash", "-c", f"cd {REMOTE} && tar -czf prepare-artifacts.tar.gz --ignore-failed-read trusted *.log"],
                        timeout=120,
                    )
                    if archive.success:
                        await env.download(f"{REMOTE}/prepare-artifacts.tar.gz", str(folder / "prepare-artifacts.tar.gz"))
                        with tarfile.open(folder / "prepare-artifacts.tar.gz") as retained:
                            retained.extractall(folder, filter="data")
                    else:
                        (folder / "prepare-artifacts-error.txt").write_text(archive.stderr)
                except Exception as error:
                    (folder / "prepare-artifacts-error.txt").write_text(repr(error))
                # Completed measurements are reusable even when a later preparation step fails.
                provenance = {"gpu_inventory": inventory, "provider": state.metadata["gpu_provider"], "sample": folder.name}
                speed_folder = folder / "trusted" / "speed"
                if cached is None and all((speed_folder / name).is_file() for name in SPEED_BASELINE_FILES):
                    try:
                        cached = store_measurement(speed_baseline_folder(identity), speed_folder, identity, provenance)
                    except Exception as error:
                        (folder / "speed-baseline-store-error.txt").write_text(repr(error))
                quality_folder = folder / "trusted" / "quality"
                if cached_quality is None and all((quality_folder / name).is_file() for name in QUALITY_REFERENCE_FILES) and registry_file(quality_folder):
                    try:
                        measured = {}
                        if (folder / "trusted" / "provenance.json").is_file():
                            measured = json.loads((folder / "trusted" / "provenance.json").read_text()).get("quality_reference", {})
                        cached_quality = store_measurement(quality_reference_folder(quality_identity), quality_folder, quality_identity, {
                            **provenance, **{key: measured.get(key) for key in ("vllm_version", "concurrency", "retried_requests", "complete")},
                        })
                    except Exception as error:
                        (folder / "quality-reference-store-error.txt").write_text(repr(error))
        (folder / "prepare.log").write_text(output)

        # Trusted scoring inputs now live outside the agent sandbox.
        state.metadata["provenance"] = json.loads((folder / "trusted" / "provenance.json").read_text())
        store().set("workspace_env", json.loads((folder / "trusted" / "environment.json").read_text()))
        state.metadata["speed_baseline"] = {
            "source": "cache" if state.metadata["cached_speed_baseline"] else "measured",
            "folder": str(cached.resolve()) if cached is not None else None,
        }
        state.metadata["quality_reference"] = {
            "source": "cache" if state.metadata["cached_quality_reference"] else "measured",
            "folder": str(cached_quality.resolve()) if cached_quality is not None else None,
        }

        seconds = state.metadata["agent_seconds"]
        deadline = time.time() + seconds if seconds is not None else None
        store().set("deadline", deadline)
        if deadline is not None:
            # Upstream's timer script prints the remaining budget in hours and minutes.
            await checked_exec(env, [
                "bash", f"{UPSTREAM_ROOT}/src/utils/create_timer.sh", num_hours_text(seconds), "/home/agent/task/timer.sh",
            ], 30)
        else:
            await env.write_file(
                "/home/agent/task/timer.sh",
                '#!/bin/sh\necho "No wall-clock limit; use the token-budget reminders."\n',
            )
            await checked_exec(env, ["chmod", "+x", "/home/agent/task/timer.sh"], 30)
        return state

    return solve


async def restart_for_scoring(state, include_transcript: bool):
    """Restore the submitted filesystem into a new H100 sandbox and reinstall trusted evaluator inputs from the host."""
    env = await gpu_environment().restart(state.metadata["gpu_config"])
    try:
        state.metadata["scoring_sandbox_id"] = env.resource_id
        folder = Path(store().get("artifacts"))
        (folder / "scoring-sandbox.json").write_text(
            json.dumps(
                {
                    "sandbox_id": env.resource_id,
                    "provider": state.metadata["gpu_provider"],
                }
            )
        )
        # Replace whatever the agent left under the evaluator paths with the pristine upstream copy and measurements.
        await install_upstream(env, state.metadata)
        await env.write_file(f"{REMOTE}/trusted.tar.gz", archive_bytes(folder / "trusted", "trusted"))
        await checked_exec(env, ["tar", "-xzf", f"{REMOTE}/trusted.tar.gz", "-C", REMOTE], 60)
        # Remove stale harness exports even when transcript review is disabled.
        await checked_exec(
            env,
            ["rm", "-f", f"{REMOTE}/final.json", f"{REMOTE}/agent-transcript.json"],
            30,
        )
        if include_transcript:
            evidence = folder / "agent-transcript.json"
            write_agent_transcript(evidence, state)
            await env.upload(str(evidence), f"{REMOTE}/agent-transcript.json")
        # After restart, background servers can no longer change the saved workspace.
        if os.environ.get("HAWK_JOB_ID"):
            await copy_submission(env, folder)
        return env
    except BaseException:
        with anyio.CancelScope(shield=True):
            await env.terminate()
        raise


async def retain_failed_submission(state):
    """Save unfinished work before Inspect removes the sandbox after a solver error."""
    folder = store().get("artifacts")
    if not os.environ.get("HAWK_JOB_ID") or not folder:
        return
    folder = Path(folder)
    if not (folder / "submission.tar.gz").exists():
        write_agent_transcript(folder / "agent-transcript.json", state)
        await copy_submission(gpu_environment(), folder)


async def copy_submission(env, folder: Path):
    """Keep archive failures as diagnostics without replacing the evaluation's outcome."""
    try:
        await checked_exec(env, ["tar", "--ignore-failed-read", "-czf", f"{REMOTE}/submission.tar.gz",
                                 "-C", "/home/agent", "task"], 300)
        await env.download(f"{REMOTE}/submission.tar.gz", str(folder / "submission.tar.gz"))
    except Exception as error:
        (folder / "submission-copy-error.txt").write_text(repr(error))


def write_agent_transcript(path: Path, state) -> None:
    """Stream model outputs and tool results so the judge retains evidence removed by compaction."""
    fields = {"event", "timestamp", "span_id", "model", "role", "output",
              "function", "arguments", "result", "error"}
    seen_tools = set()
    with path.open("w") as output:
        output.write('{"events": [\n')
        separator = ""
        for event in transcript().events:
            if event.event not in {"model", "tool"}:
                continue
            record = event.model_dump(mode="json", include=fields)
            if event.event == "model":
                # CLI results live in model inputs, and are repeated on later turns.
                record["tool_results"] = []
                for message in event.input:
                    if message.role == "tool":
                        content = message.model_dump_json(exclude={"id"})
                        fingerprint = hashlib.sha256(content.encode()).digest()
                        if fingerprint not in seen_tools:
                            seen_tools.add(fingerprint)
                            record["tool_results"].append(json.loads(content))
            output.write(separator + json.dumps(record, indent=2))
            separator = ",\n"
        output.write('\n], "messages": ')
        json.dump([m.model_dump(mode="json") for m in state.messages], output, indent=2)
        output.write("}\n")
