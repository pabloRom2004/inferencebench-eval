"""Small deployment checks that do not allocate benchmark infrastructure."""
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes
from inspect_ai.solver import generate


@task
def provider_probe() -> Task:
    """Verify the installed package and model route without allocating a GPU."""
    return Task(
        dataset=[Sample(id="greeting", input="Reply with exactly the word hello.", target="hello")],
        solver=generate(),
        scorer=includes(),
    )
