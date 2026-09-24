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
