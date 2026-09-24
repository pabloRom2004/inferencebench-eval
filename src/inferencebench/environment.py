from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shlex
import shutil
import tarfile
import time
from pathlib import Path
from typing import Any

import anyio
import httpx
import modal
from fsspec.core import url_to_fs
from inspect_ai.hooks import Hooks, hooks
from inspect_ai.log import transcript
from inspect_ai.model import ModelInfo, get_model, get_model_info, set_model_info
from inspect_ai.solver import Solver, solver
from inspect_ai.util import ResumeReport, sandbox, sandboxenv, store

from inferencebench.dataset import num_hours_text
from inferencebench.harness_default import reschedule_deadline
from inferencebench.prompts import ASSETS
from inferencebench.utils.run_config import load_config
from inferencebench.utils.sandboxes.modal import FileSystemModalSandbox
from inferencebench.utils.sandboxes.runpod import RunPodSandbox as BaseRunPodSandbox
from inferencebench.vendored import patch_digest, upstream_archive, upstream_lock

REMOTE = "/tmp/inferencebench"
EVALUATOR = "/opt/evaluator/bin/python"
UPSTREAM_ROOT = "/opt/inferencebench"
# Speed baselines taken once per workload and GPU model and shared by later samples, like upstream's precomputed registries.
BASELINE_CACHE = Path("run-artifacts") / "baselines"
SPEED_BASELINE_FILES = ("requests.jsonl", "baseline_metrics.json")
# Checkpoints keep the workspace and each native CLI's resumable session, not installed packages or weights.
# The CLIs run as root with HOME=/home/agent; Codex keeps its session inside the workspace.
CHECKPOINT_PATHS = [
    "/home/agent/task",
    "/home/agent/.claude",
    "/home/agent/.gemini",
    "/home/agent/.kimi-code",
    "/home/agent/.local/share/opencode",
]


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
            "base_model", "scenario", "eval_seed", "max_model_len", "baseline_dtype", "seeded_arrivals",
            "scenario_a_output_tokens", "retokenize_outputs",
        )},
    }


def measurement_folder(identity: dict[str, Any], prefix: str) -> Path:
    """Place each shared measurement in a folder named by its workload and a digest of the full identity."""
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    model = re.sub(r"[^A-Za-z0-9_.-]+", "_", identity["base_model"])
    return BASELINE_CACHE / f"{prefix}-{model}-{digest}"


def speed_baseline_folder(identity: dict[str, Any]) -> Path:
    """The shared folder for one scenario, evaluation seed, and identity."""
    return measurement_folder(identity, f"{identity['scenario']}-seed{identity['eval_seed']}")


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
        # A checkpoint restore replaces the store; grading keeps this attempt's own preparation.
        state.metadata["artifacts"] = str(folder.resolve())

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

        # Reuse a speed baseline already taken for this workload and model on this GPU model.
        identity = speed_baseline_identity(state.metadata, gpu_model(inventory))
        cached = cached_speed_baseline(identity)
        state.metadata["cached_speed_baseline"] = cached is not None
        await install_upstream(env, state.metadata)
        if cached is not None:
            await checked_exec(env, ["mkdir", "-p", f"{REMOTE}/cached/speed"], 30)
            for path in (cached / name for name in SPEED_BASELINE_FILES):
                await env.upload(str(path), f"{REMOTE}/cached/speed/{path.name}")

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
                # A completed speed baseline is reusable even when a later preparation step fails.
                provenance = {"gpu_inventory": inventory, "provider": state.metadata["gpu_provider"], "sample": folder.name}
                speed_folder = folder / "trusted" / "speed"
                if cached is None and all((speed_folder / name).is_file() for name in SPEED_BASELINE_FILES):
                    try:
                        cached = store_measurement(speed_baseline_folder(identity), speed_folder, identity, provenance)
                    except Exception as error:
                        (folder / "speed-baseline-store-error.txt").write_text(repr(error))
        (folder / "prepare.log").write_text(output)

        # Trusted scoring inputs now live outside the agent sandbox.
        state.metadata["provenance"] = json.loads((folder / "trusted" / "provenance.json").read_text())
        store().set("workspace_env", json.loads((folder / "trusted" / "environment.json").read_text()))
        state.metadata["speed_baseline"] = {
            "source": "cache" if state.metadata["cached_speed_baseline"] else "measured",
            "folder": str(cached.resolve()) if cached is not None else None,
        }

        if state.metadata["checkpoint"]:
            # Restic cannot save a capture path that does not exist yet.
            await checked_exec(env, ["mkdir", "-p", *CHECKPOINT_PATHS], 30)

        seconds = state.metadata["agent_seconds"]
        deadline = time.time() + seconds if seconds is not None else None
        store().set("deadline", deadline)
        await write_timer(env, seconds)
        return state

    return solve


async def write_timer(env, seconds: float | None) -> None:
    """Install upstream's timer for the remaining optimization time, or a note that only the token budget applies."""
    if seconds is not None:
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


async def record_checkpoint_time(state) -> None:
    """Store when this checkpoint was taken, so a restore can return the optimization time lost after it."""
    store().set("checkpoint_saved_at", time.time())


async def resume_environment(state, attempt) -> ResumeReport:
    """Grade with this attempt's fresh preparation and extend the deadline by the time between the last checkpoint and this restore."""
    store().set("artifacts", state.metadata["artifacts"])
    deadline, saved = store().get("deadline"), store().get("checkpoint_saved_at")
    data = {"attempt": attempt, "checkpoint_saved_at": saved, "previous_deadline": deadline}
    if attempt == "resume" and deadline is not None and saved is not None:
        # Reinstalling what the checkpoint left out happens after this point and counts against the agent.
        downtime = time.time() - saved
        deadline += downtime
        store().set("deadline", deadline)
        reschedule_deadline(deadline)
        await write_timer(gpu_environment(), max(0, deadline - time.time()))
        data.update(downtime_seconds=downtime, deadline=deadline)
    return ResumeReport(data=data)


async def restart_for_scoring(state, include_transcript: bool):
    """Restore the submitted filesystem into a new H100 sandbox and reinstall trusted evaluator inputs from the host."""
    current = gpu_environment()
    folder = Path(store().get("artifacts"))
    if os.environ.get("HAWK_JOB_ID"):
        # Retain the submission even if snapshotting or restoring the full filesystem fails.
        if include_transcript:
            write_agent_transcript(folder / "agent-transcript.json", state)
        await copy_submission(current, folder)
    env = await current.restart(state.metadata["gpu_config"])
    try:
        state.metadata["scoring_sandbox_id"] = env.resource_id
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
        # The full filesystem snapshot should not contain a second copy of the submission.
        await checked_exec(env, ["rm", "-f", f"{REMOTE}/submission.tar.gz"], 30)
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


@sandboxenv(name="inferencebench_modal")
class InferenceSandbox(FileSystemModalSandbox):
    """Restart the submitted filesystem on a fresh GPU for trusted scoring."""

    @classmethod
    async def sample_init(cls, task_name, config, metadata):
        """Build the packaged H100 image unless the task supplied its own provider configuration."""
        return await super().sample_init(task_name, config or str(ASSETS / "sandboxes" / "compose.yaml"), metadata)

    async def restart(self, config_file: str | None) -> InferenceSandbox:
        """Preserve the submitted filesystem and allocate a fresh H100 without its running processes, keeping this registered object."""
        image = await self.sandbox.snapshot_filesystem.aio(timeout=55)
        await self.terminate()
        config = load_config(config_file or "assets/sandboxes/compose.yaml")
        resources = config["services"]["default"]
        app = await modal.App.lookup.aio(
            "inferencebench-scoring", create_if_missing=True
        )
        remote = await modal.Sandbox.create.aio(
            "sleep",
            "infinity",
            app=app,
            image=image,
            gpu=config["x-modal"]["gpu"],
            cpu=resources["cpus"],
            memory=int(resources["mem_limit"].removesuffix("g")) * 1024,
            timeout=config["x-modal"]["timeout"],
            workdir=resources["working_dir"],
        )
        # Inspect SWE resolves the sample's sandbox by name, so the judge must find the replacement here.
        self.sandbox = remote
        return self


@sandboxenv(name="inferencebench_runpod")
class RunPodSandbox(BaseRunPodSandbox):
    """Bootstrap and restart the benchmark's workload on its persistent GPU volume."""

    default_config = Path(__file__).parent / "assets/sandboxes/runpod.yaml"
    working_dir = "/home/agent/task"
    environment_file = "/etc/inferencebench-env.sh"
    boot_file = "/run/inferencebench-boot-id"
    name_prefix = "inferencebench-"
    ssh_public_key_env = "INFERENCEBENCH_SSH_PUBLIC_KEY"
    ssh_host_key_env = "INFERENCEBENCH_SSH_HOST_KEY"

    def _startup_command(self) -> str:
        """Install the Dockerfile's shared environment on first boot, or restore it after restart."""
        dockerfile = (ASSETS.parent / "Dockerfile").read_text().replace("\\\n", "")
        exports = [
            "export " + line[4:]
            for line in dockerfile.splitlines()
            if line.startswith("ENV ")
        ]
        script = "set -e\n" + "\n".join(exports) + "\n"
        if self.config["bootstrap"]:
            script += "if [ ! -f /workspace/.inferencebench/ready ]; then\n"
            script += (
                "bash -c "
                + shlex.quote((ASSETS / "scripts" / "setup_environment.sh").read_text())
                + "\n"
            )
            script += "elif ! command -v rsync >/dev/null || ! command -v sshd >/dev/null; then\n"
            script += "apt-get update && apt-get install -y rsync openssh-server python3\nfi\n"
        script += "mkdir -p /opt\nprintf %s " + shlex.quote(
            (ASSETS / "scripts" / "runpod_start.sh").read_text()
        )
        script += " > /opt/runpod_start.sh\nexec bash /opt/runpod_start.sh"
        return script


    @staticmethod
    def _configuration(path) -> dict:
        """Resolve the complete provider YAML and reject missing prerequisites before a paid API call."""
        config = load_config(path or "assets/sandboxes/runpod.yaml")
        # Existing complete provider configs predate the transport keepalive controls.
        defaults = load_config("assets/sandboxes/runpod.yaml")
        for name in ["ssh_keepalive_interval_seconds", "ssh_keepalive_count_max"]:
            config.setdefault(name, defaults[name])
        config["pod"]["imageName"] = (
            os.environ.get("RUNPOD_IMAGE") or config["pod"]["imageName"]
        )
        if not os.environ.get("RUNPOD_API_KEY") or not config["pod"]["imageName"]:
            raise ValueError(
                "RunPod requires RUNPOD_API_KEY and an image in RUNPOD_IMAGE or pod.imageName"
            )
        if type(config["bootstrap"]) is not bool:
            raise ValueError("bootstrap must be true or false")
        if (
            config["pod"]["volumeMountPath"] != "/workspace"
            or config["pod"]["volumeInGb"] <= 0
        ):
            raise ValueError(
                "RunPod requires a persistent volume at /workspace for the scoring restart"
            )
        for name in [
            "api_timeout_seconds",
            "api_retry_attempts",
            "ssh_keepalive_interval_seconds",
            "ssh_keepalive_count_max",
            "startup_timeout_seconds",
            "snapshot_timeout_seconds",
            "poll_interval_seconds",
            "create_retry_attempts",
            "create_retry_interval_seconds",
        ]:
            if type(config[name]) is not int or config[name] <= 0:
                raise ValueError(f"{name} must be a positive integer")
        return config


    async def restart(self, config_file: str | None) -> RunPodSandbox:
        """Snapshot the installed filesystem to the volume and reboot the pod before grading."""
        boot = await self.read_file("/run/inferencebench-boot-id")
        await self.write_file(
            "/opt/runpod_start.sh", (ASSETS / "scripts" / "runpod_start.sh").read_text()
        )
        result = await self.exec(
            ["bash", "/opt/runpod_start.sh", "snapshot"],
            timeout=self.config["snapshot_timeout_seconds"],
        )
        if not result.success:
            raise RuntimeError(f"RunPod filesystem snapshot failed: {result.stderr}")
        for attempt in range(self.config["api_retry_attempts"]):
            self._record("restarting", restart_attempt=attempt + 1)
            try:
                await self._request("POST", f"/pods/{self.pod_id}/restart")
            except (httpx.TransportError, httpx.HTTPStatusError) as error:
                if (
                    isinstance(error, httpx.HTTPStatusError)
                    and error.response.status_code != 429
                    and error.response.status_code < 500
                ):
                    raise
                self._record("restart_response_failed", restart_error=repr(error))
                # A failed response may follow an accepted restart. Allow the full
                # restoration window before considering another restart request.
                try:
                    await self._wait_ready(boot.strip())
                    return self
                except TimeoutError:
                    if attempt + 1 == self.config["api_retry_attempts"]:
                        raise error
            else:
                await self._wait_ready(boot.strip())
                return self
        return self
