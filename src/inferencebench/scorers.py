import asyncio
import importlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import anyio
import inspect_swe
from inspect_ai.agent import AgentState
from inspect_ai.model import ChatMessageUser, GenerateConfig, get_model, model_roles
from inspect_ai.scorer import Score, Scorer, scorer
from inspect_ai.util import store

from inferencebench.dataset import SCENARIOS
from inferencebench.environment import REMOTE, checked_exec, restart_for_scoring
from inferencebench.metrics import (
    aggregate_speedup,
    performance,
    scored_attempts,
    unscored_attempts,
)
from inferencebench.prompts import JUDGE_TRANSCRIPT, Prompt, select_prompt
from inferencebench.utils.run_config import load_config
from inferencebench.vendored import UPSTREAM

DEFAULT_SCORER_ARGS = load_config()["task"]["args"]["scorer"]["args"]

# Upstream's judge prompt builder inlines these two files from the task directory, in this order.
EVIDENCE_FILES = [
    ("start_server.sh", "/home/agent/task/start_server.sh"),
    ("server.log", f"{REMOTE}/final-server.log"),
]
JUDGE_PROMPT_SCRIPT = UPSTREAM / "src/disallowed_usage_judge/get_judge_prompt.py"
# Upstream's judge writes its two verdicts as files in the submission directory.
VERDICT_FILES = ["contamination_judgement.txt", "disallowed_model_judgement.txt"]
# The judge runs the same CLI upstream used; bound here so the judge can be replaced independently of the agent.
judge_cli = inspect_swe.claude_code


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


async def stage_evidence(env, folder: Path) -> list[str]:
    """Copy the restarted submission's launcher and final server log where upstream's prompt builder expects them."""
    evidence = folder / "judge-evidence"
    evidence.mkdir(exist_ok=True)
    names = []
    for filename, path in EVIDENCE_FILES:
        try:
            content = await env.read_file(path)
        except Exception:
            # Upstream silently skips unreadable evidence files.
            continue
        (evidence / filename).write_text(content)
        names.append(filename)
    return names


async def upstream_judge_prompt(rubric: Prompt, benchmark: str, model: str, task_dir: Path) -> str:
    """Render the rubric and pre-loaded evidence with upstream's own get_judge_prompt.py."""
    env = {key: value for key, value in os.environ.items() if key not in {"INFERENCE_BENCH_PROMPT", "POST_TRAIN_BENCH_PROMPT"}}
    with tempfile.TemporaryDirectory() as staging:
        cwd = UPSTREAM
        if rubric.prompt != (UPSTREAM / "src/disallowed_usage_judge/prompt.txt").read_text():
            # The script reads its rubric from a fixed relative path; an alternate rubric is staged there.
            cwd = Path(staging)
            (cwd / "src/disallowed_usage_judge").mkdir(parents=True)
            (cwd / "src/disallowed_usage_judge/prompt.txt").write_text(rubric.prompt)
        process = await asyncio.create_subprocess_exec(
            sys.executable, str(JUDGE_PROMPT_SCRIPT), "--benchmark", benchmark, "--model", model, "--task-dir", str(task_dir),
            cwd=cwd, env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"Upstream judge prompt builder failed: {stderr.decode()}")
    # Upstream captures the script's output with command substitution, which drops trailing newlines.
    return stdout.decode().rstrip("\n")


async def judge_submission(
    env,
    state,
    grader_role: str,
    judge_prompt: str,
    max_grader_attempts: int,
    include_transcript: bool,
    transcript_hint: bool,
    judge_cli_version: str,
) -> tuple[bool | None, str, dict[str, Any]]:
    """Run the integrity judge the way upstream does: Claude Code in the submission directory, verdicts written as files."""
    folder = Path(store().get("artifacts"))
    evidence_names = await stage_evidence(env, folder)
    prompt = await upstream_judge_prompt(
        select_prompt(judge_prompt, "judge_prompt"),
        SCENARIOS[state.metadata["scenario"]]["benchmark"],
        state.metadata["base_model"],
        folder / "judge-evidence",
    )
    if transcript_hint:
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
    # The same CLI, working directory, and permission bypass as upstream's `claude --print` call.
    judge = judge_cli(
        model=model,
        version=judge_cli_version,
        cwd="/home/agent/task",
        user="root",
        sandbox="default",
        permission_mode="bypassPermissions",
        retry_refusals=0,
    )
    paths = [f"/home/agent/task/{name}" for name in VERDICT_FILES]

    result = ""
    for attempt_number in range(1, max_grader_attempts + 1):
        # Verdicts must come from this judge run, not from files the agent left behind.
        await checked_exec(env, ["rm", "-f", *paths], 30)
        attempt = await judge(AgentState(messages=[ChatMessageUser(content=prompt)]))
        result = attempt.messages[-1].text if attempt.messages else ""
        verdicts = {}
        for name, path in zip(VERDICT_FILES, paths):
            try:
                verdicts[name] = await env.read_file(path)
            except Exception:
                # Upstream only warns when the judge leaves a verdict file unwritten.
                continue
        verdict = parse_judgment("\n".join(verdicts.values()))
        details = {
            "model": str(model),
            "role": grader_role,
            "attempts": attempt_number,
            "include_transcript": include_transcript,
            "transcript_hint": transcript_hint,
            "preloaded_evidence": evidence_names,
            "judge_cli_version": judge_cli_version,
            "verdict_files": verdicts,
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
    transcript_hint: bool = DEFAULT_SCORER_ARGS["transcript_hint"],
    judge_cli_version: str = DEFAULT_SCORER_ARGS["judge_cli_version"],
) -> Scorer:
    """Score held-out inference speed relative to Transformers after restart, quality, and integrity checks."""
    select_prompt(judge_prompt, "judge_prompt")
    if type(max_grader_attempts) is not int or max_grader_attempts < 1:
        raise ValueError("max_grader_attempts must be a positive integer")
    for name, value in [("include_transcript", include_transcript), ("transcript_hint", transcript_hint)]:
        if type(value) is not bool:
            raise ValueError(f"{name} must be a boolean")
    if transcript_hint and not include_transcript:
        raise ValueError("transcript_hint requires include_transcript")
    if type(judge_cli_version) is not str or not judge_cli_version or judge_cli_version in {"auto", "latest", "stable", "sandbox"}:
        raise ValueError("judge_cli_version must be an exact Claude Code version")

    async def score(state, target):
        """Retain raw final measurements and distinguish invalid submissions from unavailable judgments."""
        folder = Path(store().get("artifacts"))
        baseline = json.loads((folder / "trusted" / "speed" / "baseline_metrics.json").read_text())["baseline"]
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
                transcript_hint,
                judge_cli_version,
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
