import json
from typing import Any

from inspect_ai.dataset import MemoryDataset, Sample

from inferencebench.prompts import ASSETS, STRICT_RULES, select_prompt

SCENARIOS = json.loads((ASSETS / "datasets" / "scenarios.json").read_text())


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

    # Render the upstream prompt, then vary only the workload seeds.
    samples = []
    for scenario in selected:
        record = SCENARIOS[scenario]
        prompt = select_prompt(options["system_prompt"], "system_prompt").prompt
        if options["strict_prompt"]:
            anchor = "* Base Model: You must use {model}."
            if anchor not in prompt:
                raise ValueError("strict_prompt requires the original base-model constraint in the selected prompt")
            line_end = prompt.index("\n", prompt.index(anchor)) + 1
            prompt = prompt[:line_end] + STRICT_RULES.prompt + "\n" + prompt[line_end:]
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
