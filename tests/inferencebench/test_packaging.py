from pathlib import Path
from unittest.mock import Mock

import modal
from inspect_ai.util import ComposeConfig
from inspect_sandboxes.modal._compose import convert_compose_to_modal_params

from inferencebench import inference_bench
from inferencebench.utils.run_config import load_config


def test_modal_build_context_retains_bootstrap_assets(monkeypatch, tmp_path):
    """Resolve the packaged Modal build and its copied assets outside the checkout."""
    task = inference_bench()
    compose_path = task.sandbox.config
    image = Mock()
    build = Mock(return_value=image)
    monkeypatch.setattr(modal.Image, "from_dockerfile", build)
    monkeypatch.chdir(tmp_path)
    result = convert_compose_to_modal_params(
        ComposeConfig.model_validate(load_config(compose_path)), compose_path
    )
    assert result.kwargs["image"] is image
    dockerfile = Path(build.call_args.args[0])
    context = Path(build.call_args.kwargs["context_dir"])
    assert dockerfile.is_file()
    for line in dockerfile.read_text().splitlines():
        if line.startswith("COPY "):
            assert (context / line.split()[1]).is_file()


def test_provider_probe_has_no_gpu_or_benchmark_workload(tmp_path):
    """Run the deployment probe through Inspect while keeping GPU allocation absent."""
    from inspect_ai import eval as inspect_eval
    from inspect_ai.model import ModelOutput, get_model

    from inferencebench import provider_probe

    task = provider_probe()
    assert task.sandbox is None
    [log] = inspect_eval(
        task,
        model=get_model("mockllm/probe", custom_outputs=[ModelOutput.from_content("mockllm/probe", "hello")]),
        log_dir="logs",
        display="none",
    )
    try:
        assert log.status == "success"
        assert log.samples[0].scores["includes"].value == "C"
    finally:
        Path(log.location).unlink()
