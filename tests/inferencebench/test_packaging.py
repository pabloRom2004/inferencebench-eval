from pathlib import Path
from unittest.mock import Mock

import modal
from inspect_ai.util import ComposeConfig
from inspect_sandboxes.modal._compose import convert_compose_to_modal_params

from inferencebench import inference_bench
from inferencebench.prompts import ASSETS
from inferencebench.utils.run_config import load_config


def test_modal_build_context_retains_bootstrap_assets(monkeypatch, tmp_path):
    """Resolve the packaged Modal build and its copied assets outside the checkout."""
    assert inference_bench().sandbox.config is None
    compose_path = str(ASSETS / "sandboxes" / "compose.yaml")
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


async def test_modal_defaults_to_packaged_compose(monkeypatch):
    """Allocate Modal samples from the packaged compose file when the task logs no provider configuration."""
    from inferencebench.environment import InferenceSandbox
    from inferencebench.utils.sandboxes.modal import FileSystemModalSandbox

    seen = []

    async def sample_init(cls, task_name, config, metadata):
        """Record the configuration the Modal provider would build from."""
        seen.append(config)
        return {}

    monkeypatch.setattr(FileSystemModalSandbox, "sample_init", classmethod(sample_init))
    await InferenceSandbox.sample_init("task", None, {})
    await InferenceSandbox.sample_init("task", "/custom/compose.yaml", {})
    assert seen == [str(ASSETS / "sandboxes" / "compose.yaml"), "/custom/compose.yaml"]
