from dataclasses import dataclass
from pathlib import Path

from inferencebench.vendored import UPSTREAM as UPSTREAM_TREE
from inferencebench.vendored import upstream_lock

ASSETS = Path(__file__).parent / "assets"
UPSTREAM_COMMIT = upstream_lock()["commit"]
UPSTREAM = f"{upstream_lock()['source']}/blob/{UPSTREAM_COMMIT}"


@dataclass(frozen=True)
class Prompt:
    """A named prompt with its original source and Inspect placement."""

    title: str
    prompt: str
    role: str
    origin: str
    origin_note: str = ""
    inspect_parameter: str | None = None
    affordances: dict[str, str] | None = None
    output_format: object = None


PROMPTS = {
    "token_budget": Prompt(
        title="token_budget",
        prompt=(ASSETS / "prompts" / "token_prompt.txt").read_text(),
        role="subject",
        origin=f"{UPSTREAM}/src/eval/general/prompt.txt",
        origin_note="Original task text adapted to the configured token budget and optional optimization deadline.",
        inspect_parameter="system_prompt",
        affordances={
            "tools": "Root shell, Python, web search, Internet, and the H100 server workspace."
        },
        output_format="Standalone /home/agent/task/start_server.sh",
    ),
    "original": Prompt(
        title="original",
        prompt=(UPSTREAM_TREE / "src/eval/general/prompt.txt").read_text(),
        role="subject",
        origin=f"{UPSTREAM}/src/eval/general/prompt.txt",
        origin_note="Read from the vendored upstream copy; only the upstream placeholders are replaced.",
        inspect_parameter="system_prompt",
        affordances={
            "tools": "Root shell, coding CLI, Internet, and the H100 server workspace."
        },
        output_format="Standalone /home/agent/task/start_server.sh",
    ),
    "original_judge": Prompt(
        title="original_judge",
        prompt=(UPSTREAM_TREE / "src/disallowed_usage_judge/prompt.txt").read_text(),
        role="judge",
        origin=f"{UPSTREAM}/src/disallowed_usage_judge/prompt.txt",
        origin_note="Read from the vendored upstream copy and rendered by upstream's own get_judge_prompt.py, then given to Claude Code in the submission directory as upstream does.",
        inspect_parameter="judge_prompt",
        affordances={"claude_code": "Upstream's judge harness with its own shell and file tools in the restarted submission."},
        output_format="Two upstream verdict files in the submission directory",
    ),
}

STRICT_RULES = Prompt(
    title="strict_rules",
    prompt=(
        "* Checkpoints: Do not download or serve third-party pre-quantized, distilled, or fine-tuned checkpoints of {model}, even under its name; if you want lower precision, quantize the provided weights yourself.\n"
        "* Evaluation harness: Do not modify, replace, wrap, or bypass evaluate.py, the evaluator under /opt/inference_eval, or any task-provided requests or quality data."
    ),
    role="subject",
    origin="Inspect port; the leaderboard's strict-prompt footnote at https://inferencebench.ai (retrieved 2026-09-17)",
    origin_note="The strict prompt behind the site's dagger-marked runs is unreleased; these bullets state the footnote's two rules in the original prompt's style and are inserted after its base-model constraint.",
    inspect_parameter="strict_prompt",
)

AUTOMATED_TUNING = Prompt(
    title="automated_hyperparameter_tuning",
    prompt=(
        "When you reach the hyperparameter-tuning stage, you must use an automated optimization tool or programmatic search loop. "
        "Do not tune hyperparameters by manually choosing, editing, and testing one configuration at a time. "
        "Define a search space and a measured objective, then use a tool such as Optuna or an equivalent automated random or Bayesian search "
        "to select and evaluate multiple configurations without manually choosing each trial. "
        "Run GPU trials sequentially on your single H100. Record each configuration, its measured score, and any failures, and respect the benchmark's quality requirements. "
        "Use the remaining time to confirm the best valid configuration and leave it reproducibly encoded in start_server.sh. "
        "Manual changes are allowed to implement the search and fix errors, but must not replace automated hyperparameter search."
    ),
    role="subject",
    origin="Inspect port; user-requested automated-tuning comparison",
    origin_note="Optional instruction appended to the task prompt; it does not install or choose an optimization tool for the agent.",
    inspect_parameter="automated_tuning",
)

TOKEN_BUDGET_DESCRIPTION = Prompt(
    title="token_budget_description",
    prompt="Use the full token budget shown in the live reminders; there is no wall-clock optimization limit.",
    role="subject",
    origin="Inspect port's maintained token-budget prompt",
)

TIMED_BUDGET_DESCRIPTION = Prompt(
    title="timed_budget_description",
    prompt="You have {num_hours} hours of wall-clock optimization time, starting after environment preparation. The token limit also applies; stop when either budget ends. Use ./timer.sh to check the remaining time and leave the best working start_server.sh ready before the deadline.",
    role="subject",
    origin="Inspect port; render the configured agent_seconds deadline in the maintained prompt",
    inspect_parameter="agent_seconds",
)

ORIGINAL_CLI_CONTEXT = Prompt(
    title="original_claude_context",
    prompt="\nYou are running in a non-interactive mode. So make sure every process you are running finishes before you write your last message.\n",
    role="subject",
    origin=f"{UPSTREAM}/src/eval/general/get_prompt.py",
    origin_note="The upstream prompt builder appends this only for API-Claude, not claude_non_api.",
)

CONTINUE_PROMPT = Prompt(
    title="continue_optimization",
    prompt="Continue where you left off and complete the task. You have approximately {minutes} minutes ({seconds} seconds) of wall-clock time remaining. Make sure you utilize this time fully to achieve the best results you can. Do not ask for user feedback.",
    role="subject",
    origin=f"{UPSTREAM}/agents/claude/solve.sh",
    origin_note="Verbatim default resume message, with printf fields replaced by named placeholders.",
)


NUDGE_PROMPT = Prompt(
    title="continue_until_token_budget",
    prompt="Continue working on the inference server. Use tools to explore, implement, and measure improvements. A working deployment or a final answer does not end this attempt. Use the remaining token budget, and keep the best standalone start_server.sh ready for final evaluation.",
    role="subject",
    origin="Inspect port; requested by Pablo on 2026-09-09",
    origin_note="Unbounded early-answer continuation, following the ExploitBench default agent pattern.",
    inspect_parameter="nudge_prompt",
)

TOKEN_BUDGET_REMINDER = Prompt(
    title="token_budget_reminder",
    prompt="Token budget: {used:,.0f} / {limit:,.0f} input-plus-output tokens used; {remaining:,.0f} remaining. This counts cumulative usage across API calls, including cached input. Keep improving and measuring the server until the budget is exhausted.",
    role="subject",
    origin="Inspect port; requested by Pablo on 2026-09-09",
    origin_note="Rendered from Inspect's effective sample limit before the first turn and after each turn.",
    inspect_parameter="token_budget_reminder",
)

JUDGE_TRANSCRIPT = Prompt(
    title="inspect_judge_transcript",
    prompt="Also read {remote}/agent-transcript.json: it is the agent's full transcript exported from the Inspect eval log, with every model output and tool result, and is evidence for your judgment.",
    role="judge",
    origin="Inspect port",
    origin_note="Prompt pointer to the exported transcript, appended after upstream's rendered judge prompt; default.yaml enables it, original.yaml leaves the file unmentioned like the released harness.",
    inspect_parameter="transcript_hint",
)

# Index auxiliary prompts too, while keeping task and rubric selection distinct.
PROMPTS.update(
    {
        prompt.title: prompt
        for prompt in [
            STRICT_RULES,
            AUTOMATED_TUNING,
            TOKEN_BUDGET_DESCRIPTION,
            TIMED_BUDGET_DESCRIPTION,
            ORIGINAL_CLI_CONTEXT,
            CONTINUE_PROMPT,
            NUDGE_PROMPT,
            TOKEN_BUDGET_REMINDER,
            JUDGE_TRANSCRIPT,
        ]
    }
)


def select_prompt(title: str, parameter: str) -> Prompt:
    """Select a prompt only for the task or judge parameter it was authored to serve."""
    prompt = PROMPTS.get(title)
    if prompt is None or prompt.inspect_parameter != parameter:
        raise ValueError(f"Unknown {parameter} prompt: {title}")
    return prompt
