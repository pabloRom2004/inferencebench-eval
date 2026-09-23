import hashlib
from functools import cache
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml
from inspect_ai.util import SandboxEnvironmentSpec


@cache
def _sandbox_config_directory() -> TemporaryDirectory[str]:
    """Retain generated sandbox files for the lifetime of the controller."""
    return TemporaryDirectory(prefix="inspect-k8s-")


@cache
def kubernetes_sandbox(
    image: str, config_file: str, node_selector: tuple[tuple[str, str], ...]
) -> SandboxEnvironmentSpec:
    """Create native Kubernetes values for one digest-pinned challenge image."""
    values = yaml.safe_load(Path(config_file).read_text())
    repositories = values.pop("imageRepositories", {})
    repository, separator, digest = image.partition("@")
    image = repositories.get(repository, repository) + separator + digest
    values["services"]["default"]["image"] = image
    if node_selector:
        values["services"]["default"]["nodeSelector"] = dict(node_selector)
    rendered = yaml.safe_dump(values)
    filename = hashlib.sha256(rendered.encode()).hexdigest() + ".yaml"
    path = Path(_sandbox_config_directory().name) / filename
    path.write_text(rendered)
    return SandboxEnvironmentSpec(type="k8s", config=str(path))
