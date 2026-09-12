import math
from pathlib import Path
from typing import Any

from inspect_ai import Epochs, Task, task
from inspect_ai.agent import as_solver
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import Solver
from inspect_ai.util import SandboxEnvironmentSpec, registry_create

from inferencebench.dataset import get_inference_dataset
from inferencebench.environment import prepare_environment, retain_failed_submission
from inferencebench.quality_cache import load_quality_cache
from inferencebench.run_config import load_config
from inferencebench.scorers import scorers_from_spec

CONFIG = load_config()
DEFAULT_TASK_ARGS = CONFIG["task"]["args"]


@task
def inference_bench(
    # Packaged fallback for native Inspect settings; runtime overrides still win.
    config_defaults:         str = DEFAULT_TASK_ARGS["config_defaults"],

    # GPU provider and optional provider configuration file.
    gpu_provider:            str = DEFAULT_TASK_ARGS["gpu_provider"],
    gpu_config:              str | None = DEFAULT_TASK_ARGS["gpu_config"],

    # Dataset filters.
    scenarios:               str | list[str] | None = DEFAULT_TASK_ARGS["scenarios"],
    seed_pairs:              list[list[int]] = DEFAULT_TASK_ARGS["seed_pairs"],

    # Workload and optimization budget.
    base_model:              str = DEFAULT_TASK_ARGS["base_model"],
    max_model_len:           int = DEFAULT_TASK_ARGS["max_model_len"],
    context_length:          int | None = DEFAULT_TASK_ARGS["context_length"],
    agent_seconds:           int | None = DEFAULT_TASK_ARGS["agent_seconds"],
    request_limit:           int | None = DEFAULT_TASK_ARGS["request_limit"],
    request_cache:           str | None = DEFAULT_TASK_ARGS["request_cache"],
    quality_samples:         int = DEFAULT_TASK_ARGS["quality_samples"],
    quality_seed:            int = DEFAULT_TASK_ARGS["quality_seed"],
    quality_cache:           str | None = DEFAULT_TASK_ARGS["quality_cache"],
    quality_cache_dir:       str = DEFAULT_TASK_ARGS["quality_cache_dir"],
    quality_tau:             float = DEFAULT_TASK_ARGS["quality_tau"],
    quality_concurrency:     int = DEFAULT_TASK_ARGS["quality_concurrency"],
    quality_baseline_max_attempts: int = DEFAULT_TASK_ARGS["quality_baseline_max_attempts"],
    server_wait_seconds:     int = DEFAULT_TASK_ARGS["server_wait_seconds"],
    request_timeout_seconds: int = DEFAULT_TASK_ARGS["request_timeout_seconds"],

    # Prompt and scoring.
    system_prompt:           str = DEFAULT_TASK_ARGS["system_prompt"],
    scorer:                  dict[str, Any] = DEFAULT_TASK_ARGS["scorer"],
) -> Task:
    """Optimize Mistral inference for four workloads on one H100 using the original evaluator."""
    if config_defaults not in {"default", "original"}:
        raise ValueError("config_defaults must be default or original")
    if gpu_provider not in {"modal", "runpod"}:
        raise ValueError("gpu_provider must be modal or runpod")
    if gpu_config is not None:
        gpu_config = str(Path(gpu_config).expanduser().resolve())
    if request_cache is not None:
        request_cache = str((Path(__file__).parent / Path(request_cache).expanduser()).resolve())
    options = {
        name: value
        for name, value in locals().items()
        if name not in {"scenarios", "seed_pairs", "scorer"}
    }
    counts = [max_model_len, quality_samples, quality_concurrency, quality_baseline_max_attempts]
    if context_length is not None and (type(context_length) is not int or context_length <= 0):
        raise ValueError("context_length must be a positive integer or null")
    if request_limit is not None:
        counts.append(request_limit)
    if any(type(value) is not int or value <= 0 for value in counts):
        raise ValueError("Model length, concurrency, and sample counts must be positive integers")
    if type(quality_seed) is not int or quality_seed < 0:
        raise ValueError("quality_seed must be a nonnegative integer")
    durations = [server_wait_seconds, request_timeout_seconds]
    if agent_seconds is not None:
        durations.append(agent_seconds)
    if any(type(value) is not int or value <= 0 for value in durations):
        raise ValueError("Durations must be positive integer seconds")
    if type(quality_tau) not in {int, float} or not math.isfinite(quality_tau) or not 0 < quality_tau <= 1:
        raise ValueError("quality_tau must be finite and in (0, 1]")
    config = load_config(f"run_configs/{config_defaults}.yaml")
    reference = load_quality_cache(options)
    options["quality_cache"] = str(reference) if reference is not None else None

    # Setup and grading stay in place when a caller replaces the solver.
    return Task(
        dataset=get_inference_dataset(scenarios, seed_pairs, options),
        setup=prepare_environment(),
        cleanup=retain_failed_submission,
        solver=_solver_from_config(config["solver"]),
        scorer=scorers_from_spec(scorer),
        sandbox=SandboxEnvironmentSpec(
            f"inferencebench_{gpu_provider}",
            gpu_config or str(Path(__file__).parent / ("compose.yaml" if gpu_provider == "modal" else "runpod.yaml")),
        ),
        config=GenerateConfig(**config["generate_config"]),
        epochs=Epochs(config["eval_config"]["epochs"], config["eval_config"]["epochs_reducer"]),
        token_limit=config["eval_config"]["token_limit"],
        turn_limit=config["eval_config"].get("turn_limit"),
        time_limit=config["eval_config"]["time_limit"],
        working_limit=config["eval_config"].get("working_limit"),
        message_limit=config["eval_config"].get("message_limit"),
        cost_limit=config["eval_config"].get("cost_limit"),
        fail_on_error=config["eval_config"].get("fail_on_error"),
        continue_on_fail=config["eval_config"].get("continue_on_fail"),
        score_on_error=config["eval_config"].get("score_on_error"),
        version=load_config("eval.yaml")["version"],
        metadata=load_config("eval.yaml"),
    )


def _solver_from_config(spec: dict[str, Any]) -> Solver:
    """Construct the YAML-selected default solver while retaining native Inspect solver overrides."""
    name, args = spec["solver"], spec.get("args", {})
    try:
        return registry_create("solver", name, **args)
    except LookupError as error:
        # Registry creation can also raise errors from inside a valid solver factory.
        if str(error) != f"{name} was not found in the registry":
            raise
        return as_solver(registry_create("agent", name, **args))
