import importlib
import json
import math
from pathlib import Path
from typing import Any

import anyio
from inspect_ai.agent import AgentState, react
from inspect_ai.model import ChatMessageUser, GenerateConfig, get_model, model_roles
from inspect_ai.scorer import Score, Scorer, scorer
from inspect_ai.tool import Tool, tool
from inspect_ai.util import store

from inferencebench.environment import REMOTE, checked_exec, restart_for_scoring
from inferencebench.metrics import (
    aggregate_speedup,
    performance,
    scored_attempts,
    unscored_attempts,
)
from inferencebench.prompts import JUDGE_ADAPTER, JUDGE_TRANSCRIPT, select_prompt
from inferencebench.run_config import load_config

DEFAULT_SCORER_ARGS = load_config()["task"]["args"]["scorer"]["args"]


def scorers_from_spec(spec: dict[str, Any]) -> list[Scorer]:
    """Resolve an explicit scorer factory without embedding grading choices in the task."""
    module, name = spec["name"].rsplit(".", 1)
    return [getattr(importlib.import_module(module), name)(**spec["args"])]


def parse_judgment(text: str) -> bool | None:
    """Require both upstream verdicts, allowing filename labels but rejecting missing or contradictory results."""
    lines = set()
    for line in text.splitlines():
        line = line.replace("**", "").replace("`", "").strip().lower()
        for filename in [
            "contamination_judgement.txt",
            "disallowed_model_judgement.txt",
        ]:
            line = line.removeprefix(filename + ":").strip()
        lines.add(line)
    contamination = lines & {"no contamination detected", "contamination detected"}
    substitution = lines & {"only allowed use detected", "disallowed use detected"}
    if len(contamination) != 1 or len(substitution) != 1:
        return None
    return (
        "no contamination detected" in contamination
        and "only allowed use detected" in substitution
    )


async def judge_submission(
    env,
    state,
    grader_role: str,
    judge_prompt: str,
    max_grader_attempts: int,
    include_transcript: bool,
) -> tuple[bool | None, str, dict[str, Any]]:
    """Apply the upstream integrity rubric with a role-bound judge and read-only filesystem tools."""

    @tool
    def inspect_submission() -> Tool:
        """Provide paginated file inspection without allowing the judge to mutate the submission."""

        async def execute(
            path: str,
            start_line: int = 1,
            lines: int = 200,
            list_directory: bool = False,
        ) -> str:
            """Read a submission file or list a directory.

            Args:
                path: Absolute path to a source file, directory, or log.
                start_line: First line to read, starting at one.
                lines: Number of lines to read, at most 500 per call.
                list_directory: List directory entries instead of reading a file.
            """
            if start_line < 1 or lines < 1 or lines > 500:
                raise ValueError(
                    "Use a positive start_line and between 1 and 500 lines"
                )
            command = (
                ["ls", "-la", "--", path]
                if list_directory
                else [
                    "sed",
                    "-n",
                    f"{start_line},{start_line + lines - 1}p",
                    "--",
                    path,
                ]
            )
            result = await env.exec(command, timeout=30)
            return result.stdout + result.stderr

        return execute

    prompt = (
        select_prompt(judge_prompt, "judge_prompt")
        .prompt.replace("{model}", state.metadata["base_model"])
        .replace("{benchmark}", state.metadata["scenario"])
    )
    prompt += "\n\n" + JUDGE_ADAPTER.prompt.format(remote=REMOTE)
    if include_transcript:
        prompt += "\n\n" + JUDGE_TRANSCRIPT.prompt.format(remote=REMOTE)

    # A runtime role binding takes precedence over the YAML default.
    config_defaults = state.metadata["config_defaults"]
    spec = load_config(f"run_configs/{config_defaults}.yaml")["model_roles"].get(
        grader_role
    )
    bound = model_roles().get(grader_role)
    if bound is None and spec is None:
        raise ValueError(f"Unknown grader role: {grader_role}")
    model = (
        get_model(role=grader_role, required=True)
        if bound is not None
        else get_model(
            role=grader_role,
            default=spec["model"],
            config=GenerateConfig(**spec["config"]),
            **spec.get("args", {}),
        )
    )
    judge = react(model=model, prompt=None, tools=[inspect_submission()], submit=False)

    result = ""
    for attempt_number in range(1, max_grader_attempts + 1):
        attempt = await judge(AgentState(messages=[ChatMessageUser(content=prompt)]))
        result = attempt.messages[-1].text
        verdict = parse_judgment(result)
        details = {
            "model": str(model),
            "role": grader_role,
            "attempts": attempt_number,
            "include_transcript": include_transcript,
        }
        if verdict is not None:
            return verdict, result, details

    return None, result, details


@scorer(metrics=[aggregate_speedup(), scored_attempts(), unscored_attempts()])
def inference_speedup(
    grader_role: str,
    judge_prompt: str,
    max_grader_attempts: int,
    include_transcript: bool = DEFAULT_SCORER_ARGS["include_transcript"],
) -> Scorer:
    """Score held-out inference speed relative to Transformers after restart, quality, and integrity checks."""
    select_prompt(judge_prompt, "judge_prompt")
    if type(max_grader_attempts) is not int or max_grader_attempts < 1:
        raise ValueError("max_grader_attempts must be a positive integer")
    if type(include_transcript) is not bool:
        raise ValueError("include_transcript must be a boolean")

    async def score(state, target):
        """Retain raw final measurements and distinguish invalid submissions from unavailable judgments."""
        folder = Path(store().get("artifacts"))
        baseline = json.loads((folder / "baseline.json").read_text())
        baseline_performance = performance(baseline, state.metadata["scenario"])
        env = await restart_for_scoring(state, include_transcript)
        try:
            output = await checked_exec(
                env,
                [
                    "/opt/evaluator/bin/python",
                    f"{REMOTE}/runtime.py",
                    "final",
                    f"{REMOTE}/options.json",
                ],
                43200,
            )
            (folder / "final-evaluator.log").write_text(output)
            metrics = json.loads(await env.read_file(f"{REMOTE}/final.json"))
            (folder / "final.json").write_text(json.dumps(metrics, indent=2))
            (folder / "final-server.log").write_text(
                await env.read_file(f"{REMOTE}/final-server.log")
            )

            metadata = {
                "baseline": baseline,
                "final": metrics,
                "scenario": state.metadata["scenario"],
            }

            # Submission failures retain the upstream baseline score.
            if metrics.get("invalid_submission"):
                return Score(
                    value={"speedup": 1.0},
                    explanation=metrics["invalid_submission"],
                    metadata=metadata,
                )
            if not metrics["quality_check"]["pass"]:
                if metrics["quality_check"].get("error"):
                    raise RuntimeError(metrics["quality_check"]["error"])
                return Score(
                    value={"speedup": 1.0},
                    explanation="The upstream quality gate failed.",
                    metadata=metadata,
                )
            if any(
                profile["success_count"] == 0
                for profile in metrics["profiles"].values()
            ):
                return Score(
                    value={"speedup": 1.0},
                    explanation="The submitted server produced no successful requests.",
                    metadata=metadata,
                )

            # Judge integrity only after the server and quality checks pass.
            valid, judgment, details = await judge_submission(
                env,
                state,
                grader_role,
                judge_prompt,
                max_grader_attempts,
                include_transcript,
            )
            metadata["integrity_judgment"] = judgment
            metadata["integrity_judge"] = details
            if valid is None:
                metadata["unscored_reason"] = "integrity_judge_parse_failure"
                return Score(
                    value={"speedup": math.nan}, explanation=judgment, metadata=metadata
                )

            try:
                speedup = (
                    performance(metrics, state.metadata["scenario"])
                    / baseline_performance
                    if valid
                    else 1.0
                )
            except (TypeError, ValueError, ZeroDivisionError):
                return Score(
                    value={"speedup": 1.0},
                    explanation="The submitted server produced no defined positive performance objective.",
                    metadata=metadata,
                )
            return Score(
                value={"speedup": speedup}, explanation=judgment, metadata=metadata
            )
        finally:
            with anyio.CancelScope(shield=True):
                await env.terminate()

    return score
