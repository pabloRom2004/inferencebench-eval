"""Measure and cache a Transformers MMLU-Pro reference before running InferenceBench."""

import argparse
import asyncio
import hashlib
import json
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from inspect_ai.log import (
    EvalConfig,
    EvalDataset,
    EvalLog,
    EvalMetric,
    EvalResults,
    EvalSample,
    EvalScore,
    EvalSpec,
    EvalStats,
    write_eval_log,
)
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, ModelOutput
from inspect_ai.scorer import Score

from inferencebench.environment import REMOTE, checked_exec
from inferencebench.modal_sandbox import InferenceSandbox
from inferencebench.prompts import ASSETS
from inferencebench.quality_cache import (
    BUNDLED_QUALITY,
    QUALITY_FILES,
    load_quality_cache,
    local_quality_folder,
    quality_identity,
    read_reference,
    validate_quality_cache,
)
from inferencebench.run_config import load_config
from inferencebench.runpod_sandbox import RunPodSandbox


def write_reference_log(folder: Path, options: dict, log_dir: Path) -> Path:
    """Export the actual upstream generations as one Inspect sample per MMLU-Pro question."""
    samples, results, accuracy = read_reference(folder, options)
    provenance = json.loads((folder / "provenance.json").read_text())
    logged = []
    for sample, result in zip(samples, results, strict=True):
        messages = [
            (ChatMessageSystem if message["role"] == "system" else ChatMessageUser)(content=message["content"])
            for message in sample["messages"]
        ]
        output = ModelOutput.from_content(options["base_model"], result["model_output"])
        logged.append(EvalSample(
            id=str(sample["sample_id"]), epoch=1, input=messages, target=sample["gold_answer"],
            messages=[*messages, output.message], output=output,
            scores={"mmlu_pro": Score(value=int(result["parsed_answer"] == sample["gold_answer"]),
                                      answer=result["parsed_answer"])},
            metadata={"upstream_result": result, "generation": {
                "temperature": sample["temperature"], "max_new_tokens": sample["max_new_tokens"],
            }},
        ))
    started = datetime.fromtimestamp(provenance["started_at_unix"], UTC)
    completed = datetime.fromtimestamp(provenance["completed_at_unix"], UTC)
    identifier = uuid4().hex
    log = EvalLog(
        status="success",
        eval=EvalSpec(
            created=started.isoformat(), task="mmlu_pro_reference", task_id=identifier,
            run_id=identifier, eval_id=identifier, task_version=1,
            model=options["base_model"], task_args=quality_identity(options),
            dataset=EvalDataset(name="MMLU-Pro", samples=len(samples), sample_ids=[s.id for s in logged]),
            config=EvalConfig(epochs=1),
            metadata={"source": "Imported actual upstream Transformers reference generations; no model calls during export.",
                      "provenance": provenance},
        ),
        samples=logged,
        results=EvalResults(total_samples=len(samples), completed_samples=len(samples), scores=[
            EvalScore(name="mmlu_pro", scorer="mmlu_pro", scored_samples=len(samples),
                      metrics={"accuracy": EvalMetric(name="accuracy", value=accuracy)})
        ]),
        stats=EvalStats(started_at=started.isoformat(), completed_at=completed.isoformat()),
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{started.strftime('%Y-%m-%dT%H-%M-%S')}_mmlu-pro-reference_{identifier}.eval"
    write_eval_log(log, path)
    shutil.copyfile(path, folder / "reference.eval")
    return path


def publish_reference(folder: Path, options: dict, destination: Path, log_dir: Path, *, replace: bool = False) -> Path:
    """Publish a complete reference atomically after validating its measurements and exported log."""
    provenance = json.loads((folder / "provenance.json").read_text())
    if provenance["upstream_revision"] != quality_identity(options)["upstream_revision"]:
        raise ValueError("The reference used a different upstream evaluator revision")
    if destination.exists() and not replace:
        raise FileExistsError(f"Reference already exists: {destination}; use --force to rebuild")
    _, results, accuracy = read_reference(folder, options)
    registry = json.loads((folder / "quality.json").read_text())
    registry.update({
        "base_model": options["base_model"],
        "model_revision": provenance["downloaded_model_revision"],
        "backend": "transformers",
        "dtype": "float16",
        "reference_log": "reference.eval",
    })
    registry["datasets"]["mmlu_pro"][0].update({
        "correct": sum(row["parsed_answer"] == row["gold_answer"] for row in results),
        "accuracy": accuracy,
    })
    (folder / "quality.json").write_text(json.dumps(registry, indent=2) + "\n")
    log_path = write_reference_log(folder, options, log_dir)
    manifest = {
        "identity": quality_identity(options),
        "model_revision": provenance["downloaded_model_revision"],
        "sha256": {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in QUALITY_FILES},
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    validate_quality_cache(folder, options)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".preparing-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "reference"
        staging.mkdir()
        for name in [*QUALITY_FILES, "manifest.json"]:
            shutil.copyfile(folder / name, staging / name)
        previous = Path(temporary) / "previous"
        if destination.exists():
            if not replace:
                raise FileExistsError(f"Another preparation already published {destination}")
            destination.rename(previous)
        try:
            staging.rename(destination)
        except BaseException:
            if previous.exists():
                previous.rename(destination)
            raise
    return log_path


async def measure_reference(options: dict, folder: Path) -> None:
    """Allocate one GPU for quality preparation and always clean up that run's resources."""
    provider = InferenceSandbox if options["gpu_provider"] == "modal" else RunPodSandbox
    config = options["gpu_config"] or str(ASSETS.parent / (
        "compose.yaml" if options["gpu_provider"] == "modal" else "runpod.yaml"
    ))
    task_name = "mmlu-pro-reference"
    await provider.task_init(task_name, config)
    env = None
    try:
        environments = await provider.sample_init(task_name, config, {})
        env = environments["default"]
        (folder / "sandbox.json").write_text(json.dumps({"provider": options["gpu_provider"], "sandbox_id": env.resource_id}))
        await checked_exec(env, ["mkdir", "-p", REMOTE], 30)
        await env.write_file(f"{REMOTE}/runtime.py", (ASSETS / "scripts" / "runtime.py").read_text())
        await env.write_file(f"{REMOTE}/options.json", json.dumps(options))
        try:
            await run_quality_command(env, folder)
        finally:
            # Retain diagnostics even if a reference request exhausts its retries.
            archive = await env.exec(["tar", "-czf", f"{REMOTE}/quality-baseline.tar.gz",
                                      "-C", REMOTE, "quality-baseline", "baseline-server.log", "quality-preparation.log"], timeout=60)
            if archive.success:
                await env.download(f"{REMOTE}/quality-baseline.tar.gz", str(folder / "quality-baseline.tar.gz"))
        for name in ["quality-samples.jsonl", "quality.json", "resolved_generations.jsonl", "provenance.json"]:
            await env.download(f"{REMOTE}/{name}", str(folder / name))
    finally:
        try:
            if env is not None:
                await env.terminate()
        finally:
            await provider.task_cleanup(task_name, config, True)


async def run_quality_command(env, folder: Path) -> None:
    """Stream upstream progress to the console and retain diagnostics while the GPU command runs."""
    operation = asyncio.create_task(env.exec([
        "bash", "-c",
        f"/opt/evaluator/bin/python -u {REMOTE}/runtime.py quality-reference {REMOTE}/options.json "
        f"> {REMOTE}/quality-preparation.log 2>&1",
    ], timeout=43200, timeout_retry=False))
    cursor = 0
    try:
        while True:
            done, _ = await asyncio.wait({operation}, timeout=30)
            try:
                output = await env.read_file(f"{REMOTE}/quality-preparation.log")
            except FileNotFoundError:
                if done:
                    operation.result()
                    raise
                continue
            (folder / "prepare.log").write_text(output)
            print(output[cursor:], end="", flush=True)
            cursor = len(output)
            if done:
                result = operation.result()
                if not result.success:
                    raise RuntimeError(f"MMLU-Pro reference preparation failed ({result.returncode}): {output}")
                return
    finally:
        operation.cancel()
        await asyncio.gather(operation, return_exceptions=True)


def main() -> None:
    """Prepare the selected configuration once, reusing an existing valid reference on subsequent calls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", type=Path)
    parser.add_argument("--force", action="store_true", help="Remeasure and replace a reference only after preparation succeeds.")
    parser.add_argument("--bundle", action="store_true", help="Publish only the built-in model/seed/sample settings into package assets.")
    defaults = load_config()["task"]["args"]
    keys = ["base_model", "max_model_len", "quality_seed", "quality_samples", "quality_concurrency",
            "quality_baseline_max_attempts", "request_timeout_seconds", "server_wait_seconds",
            "quality_cache_dir", "gpu_provider", "gpu_config"]
    for key in keys:
        parser.add_argument("--" + key.replace("_", "-"), type=int if type(defaults[key]) is int else str)
    args = parser.parse_args()
    options = dict(defaults)
    if args.run_config:
        options.update(load_config(str(args.run_config.resolve()))["task"]["args"])
    options.update({key: getattr(args, key) for key in keys if getattr(args, key) is not None})
    # Use the public parameter validation without requiring either prepared dataset.
    from inferencebench.task import inference_bench
    validation = {**options, "quality_cache": None, "request_cache": None}
    options = inference_bench(**validation).dataset[0].metadata
    destination = local_quality_folder(options)
    if args.bundle:
        if quality_identity(options) != quality_identity(defaults):
            parser.error("Only the built-in Mistral model, seed, sample count, and context can be bundled")
        destination = BUNDLED_QUALITY
    if destination.exists() and not args.force:
        validate_quality_cache(destination, options)
        print(f"Using existing reference: {destination}")
        return
    if not args.bundle and not args.force:
        try:
            existing = load_quality_cache({**options, "quality_cache": "auto"})
        except ValueError:
            existing = None
        if existing:
            print(f"Using existing reference: {existing}")
            return
    folder = Path("run-artifacts") / f"mmlu-pro-reference-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-{uuid4().hex[:8]}"
    folder.mkdir(parents=True)
    (folder / "options.json").write_text(json.dumps(options, indent=2))
    print(f"Preparing {options['quality_samples']} questions for {options['base_model']}, seed {options['quality_seed']}. Artifacts: {folder.resolve()}", flush=True)
    asyncio.run(measure_reference(options, folder))
    log_path = publish_reference(folder, options, destination, Path("logs").resolve(), replace=args.force)
    print(f"Cached reference: {destination}\nInspect log: {log_path}")


if __name__ == "__main__":
    main()
