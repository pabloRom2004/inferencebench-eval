from inspect_ai.model import GenerateInput
from inspect_ai.util import sample_limits


def token_reminder(prompt: str) -> str:
    """Render the sample's cumulative token usage with the supplied task template."""
    budget = sample_limits().token
    if budget.limit is None:
        return ""
    return prompt.format(
        used=int(budget.usage),
        limit=int(budget.limit),
        remaining=max(0, budget.limit - budget.usage),
        percent=100 * budget.usage / budget.limit,
    )


def time_reminder(prompt: str) -> str:
    """Render elapsed minutes and the percentage of the sample's enforced time limit."""
    budget = sample_limits().time
    limit = budget.limit
    if limit is None or limit <= 0:
        return ""
    used = budget.usage
    return prompt.format(used=used / 60, limit=limit / 60, percent=100 * used / limit)


def summary_request(request: GenerateInput) -> bool:
    """Recognize native summary requests that must not advance task reminder counters."""
    last_user = next(
        (message.text for message in reversed(request.input) if message.role == "user"),
        "",
    )
    return (
        "Please provide your summary based on the conversation so far" in last_user
        or last_user.startswith(
            "You are about to run out of context. Create a handoff summary"
        )
    )
