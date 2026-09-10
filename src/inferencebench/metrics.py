import math
from collections import defaultdict
from statistics import fmean, geometric_mean
from typing import Any

from inspect_ai.scorer import (
    Metric,
    SampleScore,
    Score,
    ScoreReducer,
    mean_score,
    metric,
    score_reducer,
)


@metric
def aggregate_speedup() -> Metric:
    """Geometrically average scenario means, keeping seed pairs equally weighted within each scenario."""

    def calculate(scores: list[SampleScore]) -> float:
        """Return an unavailable aggregate if any constituent judgment is unavailable."""
        groups = defaultdict(list)
        for sample in scores:
            value = float(sample.score.value["speedup"])
            if not math.isfinite(value):
                return float("nan")
            groups[sample.sample_metadata["scenario"]].append(value)

        # Average seeds within scenarios before combining scenarios.
        return (
            geometric_mean([fmean(values) for values in groups.values()])
            if groups
            else float("nan")
        )

    return calculate


@metric(scores="unreduced")
def scored_attempts() -> Metric:
    """Count finite attempt scores before epoch reduction; compare against Inspect's total_samples."""

    def calculate(scores: list[SampleScore]) -> int:
        """Count numeric grades without treating unavailable judgments as completed grades."""
        return sum(
            math.isfinite(float(sample.score.value["speedup"])) for sample in scores
        )

    return calculate


@metric(scores="unreduced")
def unscored_attempts() -> Metric:
    """Count returned but unavailable judgments separately from Inspect's infrastructure errors."""

    def calculate(scores: list[SampleScore]) -> int:
        """Count NaN grades before multiple epochs are combined."""
        return sum(
            not math.isfinite(float(sample.score.value["speedup"])) for sample in scores
        )

    return calculate


def performance(metrics: dict[str, Any], scenario: str) -> float:
    """Compute the paper's scenario objective from upstream p50 latency and request throughput fields."""
    profiles = list(metrics["profiles"].values())
    if not profiles or any(p["success_count"] == 0 for p in profiles):
        raise ValueError("No successful inference requests")

    if scenario == "C":
        values = [p["request_throughput_req_per_s"] for p in profiles]
    elif scenario == "A":
        values = [1 / profiles[0]["ttft"]["p50"]]
    elif scenario == "B":
        values = [1 / profiles[0]["tpot"]["p50"]]
    else:
        values = [
            1 / profiles[0]["ttft"]["p50"],
            1 / profiles[0]["tpot"]["p50"],
            profiles[0]["request_throughput_req_per_s"],
        ]

    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Performance measurements must be finite and positive")
    return geometric_mean(values)


@score_reducer
def complete_mean() -> ScoreReducer:
    """Average independent epochs while retaining unavailable judgments as unavailable sample means."""
    base = mean_score()

    def reduce(scores: list[Score]) -> Score:
        """Preserve unavailable returned judgments; Inspect reports scoreless errored epochs separately."""
        if any(not math.isfinite(float(score.value["speedup"])) for score in scores):
            return Score(
                value={"speedup": math.nan},
                metadata={"unscored_reason": "missing_epoch_judgment"},
            )
        return base(scores)

    return reduce
