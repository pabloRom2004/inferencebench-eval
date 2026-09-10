import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

from inspect_ai.dataset import MemoryDataset, Sample

from inferencebench.prompts import ASSETS, select_prompt

SCENARIOS = json.loads((ASSETS / "datasets" / "scenarios.json").read_text())


def load_request_cache(options: dict[str, Any]) -> dict[str, Any] | None:
    """Validate the prepared workload before allocating a GPU, or select upstream sampling."""
    if options["request_cache"] is None:
        return None
    cache = json.loads(gzip.decompress(Path(options["request_cache"]).read_bytes()))
    expected = {
        "format_version": 1,
        "base_model": options["base_model"],
        "max_model_len": options["max_model_len"],
        "scenario_config_sha256": hashlib.sha256(
            json.dumps(SCENARIOS, sort_keys=True).encode()
        ).hexdigest(),
        "requests_sha256": hashlib.sha256(
            json.dumps(cache["requests"], sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest(),
    }
    for key, value in expected.items():
        if cache.get(key) != value:
            raise ValueError(
                f"Request cache has incompatible {key}; regenerate it or set request_cache: null for upstream sampling"
            )
    return cache


def cached_requests(cache, scenario: str, seed: int, limit: int | None) -> list[dict[str, Any]]:
    """Select an exact prepared prefix without silently reducing the requested workload."""
    rows = cache["requests"].get(scenario, {}).get(str(seed), [])
    count = SCENARIOS[scenario]["config"]["num_requests"] if limit is None else limit
    if not rows or count > len(rows):
        raise ValueError(
            f"Request cache has {len(rows)} rows for {scenario}/{seed}, requested {count}; "
            "regenerate it or set request_cache: null for upstream sampling"
        )
    return rows[:count]


def get_inference_dataset(
    scenarios: str | list[str] | None,
    seed_pairs: list[list[int]],
    options: dict[str, Any],
) -> MemoryDataset:
    """Build one independent optimization sample per scenario and development/evaluation seed pair."""
    selected = (
        list(SCENARIOS)
        if scenarios is None
        else ([scenarios] if isinstance(scenarios, str) else scenarios)
    )
    if (
        not selected
        or set(selected) - SCENARIOS.keys()
        or len(set(selected)) != len(selected)
    ):
        raise ValueError("scenarios must contain distinct scenario IDs from A, B, C, D")

    if not seed_pairs or any(
        len(pair) != 2 or any(type(seed) is not int or seed < 0 for seed in pair)
        for pair in seed_pairs
    ):
        raise ValueError(
            "seed_pairs must contain nonnegative [development, evaluation] integer pairs"
        )
    if len({tuple(pair) for pair in seed_pairs}) != len(seed_pairs):
        raise ValueError("seed_pairs must be distinct")

    cache = load_request_cache(options)

    # Render the upstream prompt, then vary only the workload seeds.
    samples = []
    for scenario in selected:
        record = SCENARIOS[scenario]
        prompt = select_prompt(options["system_prompt"], "system_prompt").prompt
        for key, value in {
            "model": options["base_model"],
            "scenario": record["benchmark"],
            "mission": record["mission"],
            "num_hours": options["agent_seconds"] / 3600
            if options["agent_seconds"] is not None
            else "unlimited",
            "server_url": "http://127.0.0.1:8000",
            "metrics_path": "/home/agent/task/metrics.json",
        }.items():
            prompt = prompt.replace("{" + key + "}", str(value))

        for dev_seed, eval_seed in seed_pairs:
            if cache is not None:
                for seed in {dev_seed, eval_seed}:
                    cached_requests(cache, scenario, seed, options["request_limit"])
            samples.append(
                Sample(
                    id=f"{scenario}-{dev_seed}-{eval_seed}",
                    input=prompt,
                    metadata={
                        **options,
                        "scenario": scenario,
                        "directory": record["directory"],
                        "dev_seed": dev_seed,
                        "eval_seed": eval_seed,
                    },
                )
            )

    return MemoryDataset(samples, name="InferenceBench")
