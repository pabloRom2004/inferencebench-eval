"""Locate and validate reusable Transformers MMLU-Pro reference measurements."""

import hashlib
import json
import math
import re
import shlex
from pathlib import Path
from typing import Any

from inferencebench.prompts import ASSETS
from inferencebench.run_config import load_config

BUNDLED_QUALITY = ASSETS / "datasets" / "mmlu_pro" / "mistral-7b-seed-248"
QUALITY_FILES = (
    "quality-samples.jsonl", "quality.json", "resolved_generations.jsonl",
    "reference.eval", "quality-baseline.tar.gz", "provenance.json",
)


def quality_identity(options: dict[str, Any]) -> dict[str, Any]:
    """Identify the reference's model, exact sample selection, and evaluator contract."""
    return {
        "format_version": 1,
        "upstream_revision": load_config("eval.yaml")["revision"],
        "backend": "transformers",
        "dtype": "float16",
        **{key: options[key] for key in (
            "base_model", "max_model_len", "quality_seed", "quality_samples",
        )},
    }


def local_quality_folder(options: dict[str, Any]) -> Path:
    """Place each custom reference in its own model-and-seed folder outside package assets."""
    identity = quality_identity(options)
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    model = re.sub(r"[^A-Za-z0-9_.-]+", "_", options["base_model"])
    return Path(options["quality_cache_dir"]).expanduser().resolve() / (
        f"{model}-seed-{options['quality_seed']}-n-{options['quality_samples']}-{digest}"
    )


def preparation_command(options: dict[str, Any]) -> str:
    """Provide a preparation command that reproduces the missing reference settings."""
    args = ["uv", "run", "python", "-m", "inferencebench.prepare_quality"]
    for key in (
        "base_model", "max_model_len", "quality_seed", "quality_samples",
        "quality_concurrency", "quality_baseline_max_attempts", "request_timeout_seconds",
        "server_wait_seconds", "quality_cache_dir", "gpu_provider",
    ):
        args.extend(["--" + key.replace("_", "-"), str(options[key])])
    if options["gpu_config"] is not None:
        args.extend(["--gpu-config", options["gpu_config"]])
    return shlex.join(args)


def read_reference(folder: Path, options: dict[str, Any]) -> tuple[list[dict], list[dict], float]:
    """Reject partial, mismatched, or unusable reference answers before cache publication or reuse."""
    samples = [json.loads(line) for line in (folder / "quality-samples.jsonl").read_text().splitlines()]
    results = [json.loads(line) for line in (folder / "resolved_generations.jsonl").read_text().splitlines()]
    count = options["quality_samples"]
    by_id = {str(row["sample_id"]): row for row in results}
    if (len(samples) != count or len(results) != count or len(by_id) != count
            or len({str(row["sample_id"]) for row in samples}) != count
            or set(by_id) != {str(row["sample_id"]) for row in samples}):
        raise ValueError("MMLU-Pro reference must contain every selected question exactly once")
    results = [by_id[str(sample["sample_id"])] for sample in samples]
    for sample, result in zip(samples, results, strict=True):
        if (result["success"] is not True or sample["gold_answer"] not in list("ABCDEFGHIJ")
                or result["gold_answer"] != sample["gold_answer"]):
            raise ValueError("MMLU-Pro reference contains failed requests or mismatched targets")
    accuracy = sum(row["parsed_answer"] == row["gold_answer"] for row in results) / count
    registry = json.loads((folder / "quality.json").read_text())["datasets"]["mmlu_pro"]
    if (not 0 < accuracy <= 1 or len(registry) != 1
            or registry[0]["seed"] != options["quality_seed"] or registry[0]["n"] != count
            or not math.isclose(registry[0]["accuracy"], accuracy, rel_tol=0, abs_tol=1e-12)):
        raise ValueError("MMLU-Pro reference accuracy or registry does not match its answers")
    return samples, results, accuracy


def validate_quality_cache(folder: Path, options: dict[str, Any]) -> dict[str, Any]:
    """Verify cache identity and checksums, then validate the complete measured reference."""
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["identity"] != quality_identity(options):
        raise ValueError(f"MMLU-Pro cache settings do not match: {folder}")
    if set(manifest["sha256"]) != set(QUALITY_FILES):
        raise ValueError(f"MMLU-Pro cache is incomplete: {folder}")
    for name in QUALITY_FILES:
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != manifest["sha256"][name]:
            raise ValueError(f"MMLU-Pro cache checksum mismatch: {folder / name}")
    if not manifest["model_revision"]:
        raise ValueError("MMLU-Pro cache does not identify its model revision")
    read_reference(folder, options)
    return manifest


def load_quality_cache(options: dict[str, Any]) -> Path | None:
    """Find a complete matching reference or fail before the full task allocates a GPU."""
    selection = options["quality_cache"]
    if selection is None:
        return None
    if selection == "auto":
        candidates = []
        for folder in [local_quality_folder(options), BUNDLED_QUALITY]:
            path = folder / "manifest.json"
            if not path.is_file():
                continue
            try:
                identity = json.loads(path.read_text())["identity"]
            except (ValueError, KeyError, OSError, TypeError) as error:
                raise ValueError(f"Invalid MMLU-Pro cache manifest: {path}; rebuild with --force:\n"
                                 + preparation_command(options)) from error
            if identity == quality_identity(options):
                candidates.append(folder)
                break
    else:
        candidates = [Path(selection).expanduser().resolve()]
    if not candidates:
        raise ValueError(
            f"No precomputed MMLU-Pro reference for {options['base_model']}, "
            f"seed {options['quality_seed']}, {options['quality_samples']} questions. "
            "Run these questions first, then launch the full evaluation again:\n"
            + preparation_command(options)
        )
    folder = candidates[0]
    try:
        validate_quality_cache(folder, options)
    except (ValueError, KeyError, OSError, TypeError) as error:
        selection_hint = (
            "\nThen set quality_cache: auto to select the newly prepared local reference."
            if selection != "auto" else ""
        )
        raise ValueError(f"Invalid MMLU-Pro reference at {folder}: {error}. "
                         "Rebuild it with:\n" + preparation_command(options) + " --force"
                         + selection_hint) from error
    return folder
