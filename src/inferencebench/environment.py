import json
import time
from pathlib import Path

import anyio
from inspect_ai.log import transcript
from inspect_ai.model import ModelInfo, get_model, get_model_info, set_model_info
from inspect_ai.solver import Solver, solver
from inspect_ai.util import sandbox, store

from inferencebench.dataset import cached_requests, load_request_cache
from inferencebench.modal_sandbox import InferenceSandbox
from inferencebench.prompts import ASSETS
from inferencebench.quality_cache import load_quality_cache, validate_quality_cache
from inferencebench.runpod_sandbox import RunPodSandbox

REMOTE = "/tmp/inferencebench"


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

        await checked_exec(env, ["mkdir", "-p", REMOTE], 30)
        await env.write_file(
            f"{REMOTE}/runtime.py", (ASSETS / "scripts" / "runtime.py").read_text()
        )
        reference = load_quality_cache(state.metadata)
        if reference is not None:
            state.metadata["quality_cache_provenance"] = validate_quality_cache(reference, state.metadata)
            for name in ["quality-samples.jsonl", "quality.json"]:
                await env.upload(str(reference / name), f"{REMOTE}/{name}")
        cache = load_request_cache(state.metadata)
        if cache is not None:
            state.metadata["request_cache_provenance"] = {
                key: value for key, value in cache.items() if key != "requests"
            }
            for name, seed in [
                ("dev", state.metadata["dev_seed"]),
                ("heldout", state.metadata["eval_seed"]),
            ]:
                rows = cached_requests(
                    cache, state.metadata["scenario"], seed, state.metadata["request_limit"]
                )
                await env.write_file(
                    f"{REMOTE}/{name}-requests.jsonl",
                    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                )
        await env.write_file(f"{REMOTE}/options.json", json.dumps(state.metadata))

        # Build reference measurements before the agent clock starts.
        try:
            output = await checked_exec(
                env,
                [
                    "/opt/evaluator/bin/python",
                    f"{REMOTE}/runtime.py",
                    "prepare",
                    f"{REMOTE}/options.json",
                ],
                43200,
            )
        finally:
            # Preserve failed attempts even when preparation aborts before optimization.
            with anyio.CancelScope(shield=True):
                try:
                    if reference is None:
                        archive = await env.exec(["tar", "-czf", f"{REMOTE}/quality-baseline.tar.gz",
                                                  "-C", REMOTE, "quality-baseline"], timeout=60)
                        if archive.success:
                            await env.download(f"{REMOTE}/quality-baseline.tar.gz", str(folder / "quality-baseline.tar.gz"))
                        else:
                            (folder / "quality-baseline-copy-error.txt").write_text(archive.stderr)
                    else:
                        (folder / "quality-cache.json").write_text(json.dumps(state.metadata["quality_cache_provenance"], indent=2))
                except Exception as error:
                    (folder / "quality-baseline-copy-error.txt").write_text(repr(error))
        (folder / "prepare.log").write_text(output)

        # Keep trusted scoring inputs outside the agent sandbox.
        for name in [
            "baseline.json",
            "quality.json",
            "heldout-requests.jsonl",
            "quality-samples.jsonl",
            "provenance.json",
        ]:
            content = await env.read_file(f"{REMOTE}/{name}")
            (folder / name).write_text(content)
        state.metadata["provenance"] = json.loads(
            (folder / "provenance.json").read_text()
        )
        await checked_exec(
            env,
            [
                "tar",
                "--exclude=baselines",
                "-czf",
                f"{REMOTE}/evaluator.tar.gz",
                "-C",
                "/opt/inferencebench",
                "src/eval",
            ],
            60,
        )
        await env.download(
            f"{REMOTE}/evaluator.tar.gz", str(folder / "evaluator.tar.gz")
        )

        seconds = state.metadata["agent_seconds"]
        deadline = time.time() + seconds if seconds is not None else None
        store().set("deadline", deadline)
        timer = (
            f'#!/bin/sh\necho "$(( {int(deadline)} - $(date +%s) )) seconds remaining"\n'
            if deadline is not None
            else '#!/bin/sh\necho "No wall-clock limit; use the token-budget reminders."\n'
        )
        await env.write_file("/home/agent/task/timer.sh", timer)
        await checked_exec(env, ["chmod", "+x", "/home/agent/task/timer.sh"], 30)
        return state

    return solve


async def restart_for_scoring(state, include_transcript: bool):
    """Restore the submitted filesystem into a new H100 sandbox and overwrite evaluator inputs from the host."""
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
        # Restore the evaluator and held-out inputs from the host.
        await env.upload(str(folder / "evaluator.tar.gz"), f"{REMOTE}/evaluator.tar.gz")
        await checked_exec(
            env,
            ["tar", "xzf", f"{REMOTE}/evaluator.tar.gz", "-C", "/opt/inferencebench"],
            60,
        )
        await checked_exec(
            env,
            [
                "cp",
                "/opt/inferencebench/src/eval/inference/bin/launch_supervised_server.sh",
                "/opt/inference_eval/bin/launch_supervised_server.sh",
            ],
            30,
        )
        await env.write_file(
            f"{REMOTE}/runtime.py", (ASSETS / "scripts" / "runtime.py").read_text()
        )
        await env.write_file(f"{REMOTE}/options.json", json.dumps(state.metadata))
        for name in ["quality.json", "heldout-requests.jsonl", "quality-samples.jsonl"]:
            await env.write_file(f"{REMOTE}/{name}", (folder / name).read_text())
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
        return env
    except BaseException:
        with anyio.CancelScope(shield=True):
            await env.terminate()
        raise


def write_agent_transcript(path: Path, state) -> None:
    """Stream model outputs and tool results so the judge retains evidence removed by compaction."""
    fields = {"event", "timestamp", "span_id", "model", "role", "output",
              "function", "arguments", "result", "error"}
    with path.open("w") as output:
        output.write('{"events": [\n')
        separator = ""
        for event in transcript().events:
            if event.event not in {"model", "tool"}:
                continue
            output.write(separator + event.model_dump_json(include=fields, indent=2))
            separator = ",\n"
        output.write('\n], "messages": ')
        json.dump([m.model_dump(mode="json") for m in state.messages], output, indent=2)
        output.write("}\n")
