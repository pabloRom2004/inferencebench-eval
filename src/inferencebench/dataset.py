from typing import Any

from inspect_ai.dataset import MemoryDataset, Sample

from inferencebench.prompts import STRICT_RULES, select_prompt
from inferencebench.vendored import UPSTREAM, scenario_directories

# Scenario names and missions come from the vendored task directories, as upstream's prompt builder reads them.
SCENARIOS = {
    scenario: {
        "directory": directory,
        "benchmark": (UPSTREAM / "src/eval/tasks" / directory / "benchmark.txt").read_text().strip(),
        "mission": (UPSTREAM / "src/eval/tasks" / directory / "mission.txt").read_text().strip(),
    }
    for scenario, directory in scenario_directories().items()
}


def num_hours_text(seconds: int) -> str:
    """Render a wall-clock budget the way upstream's submit script passes NUM_HOURS, e.g. 7200 seconds as 2."""
    return f"{seconds / 3600:g}"


def render_prompt(template: str, options: dict[str, Any], scenario: str) -> str:
    """Fill the placeholders exactly as upstream's get_prompt.py does, with its default endpoint and metrics path."""
    record = SCENARIOS[scenario]
    values = {
        "model": options["base_model"],
        "scenario": record["benchmark"],
        "mission": record["mission"],
        "num_hours": num_hours_text(options["agent_seconds"]) if options["agent_seconds"] is not None else "unlimited",
        "server_url": "http://127.0.0.1:8000",
        "metrics_path": "/home/agent/task/metrics_preview.json",
    }
    for key, value in values.items():
        template = template.replace("{" + key + "}", value)
    # Upstream captures the rendered prompt with command substitution, which drops trailing newlines.
    return template.rstrip("\n")


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
        prompt = select_prompt(options["system_prompt"], "system_prompt").prompt
        if options["strict_prompt"]:
            anchor = "* Base Model: You must use {model}."
            if anchor not in prompt:
                raise ValueError("strict_prompt requires the original base-model constraint in the selected prompt")
            line_end = prompt.index("\n", prompt.index(anchor)) + 1
            prompt = prompt[:line_end] + STRICT_RULES.prompt + "\n" + prompt[line_end:]
        prompt = render_prompt(prompt, options, scenario)

        for dev_seed, eval_seed in seed_pairs:
            samples.append(
                Sample(
                    id=f"{scenario}-{dev_seed}-{eval_seed}",
                    input=prompt,
                    metadata={
                        **options,
                        "scenario": scenario,
                        "directory": SCENARIOS[scenario]["directory"],
                        "dev_seed": dev_seed,
                        "eval_seed": eval_seed,
                    },
                )
            )

    return MemoryDataset(samples, name="InferenceBench")
