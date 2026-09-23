"""Stage verified release archives without a GitHub API lookup on every runner."""

import hashlib
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from inspect_ai.util import concurrency, sandbox

# These are release-asset digests published by the upstream GitHub repositories.
# Keep versions explicit: a new CLI release needs its own verified artifact pin.
ARCHIVES = {
    ("codex_cli", "0.154.0"): (
        "https://github.com/openai/codex/releases/download/rust-v0.154.0/"
        "codex-package-x86_64-unknown-linux-musl.tar.gz",
        "fc6e3e3b85f2cf7d664520ee5c66a7fe4aa12bae7d46834f47e2f165fd0d6f78",
    ),
    ("opencode", "1.18.31"): (
        "https://github.com/anomalyco/opencode/releases/download/v1.18.31/"
        "opencode-linux-x64-baseline.tar.gz",
        "b283e8dbe9e6fc224bb4b79992ce3bd2174b8b7b0c3e7d1b4e6024a1d11edc84",
    ),
}


async def prepare_cli_archive(harness: str, version: str | None) -> None:
    """Use Inspect SWE's pinned archive cache, verifying both cold and warm reads."""
    if (harness, version) not in ARCHIVES:
        return
    from inspect_swe._codex_cli.agentbinary import codex_cli_binary_source
    from inspect_swe._opencode.agentbinary import opencode_binary_source
    from inspect_swe._util.sandbox import detect_sandbox_platform

    platform = await detect_sandbox_platform(sandbox())
    if platform != "linux-x64":
        return
    source = (
        codex_cli_binary_source()
        if harness == "codex_cli"
        else opencode_binary_source()
    )
    assert version is not None and source.cached_package_path is not None
    path = source.cached_package_path(version, platform)
    url, checksum = ARCHIVES[(harness, version)]
    async with concurrency(f"inspect-{harness}-archive", 1, visible=False):
        await verified_archive(path, url, checksum)


async def verified_archive(path: Path, url: str, checksum: str) -> None:
    """Atomically cache an official archive only after checking its pinned digest."""
    from inspect_swe._util.download import download_file

    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == checksum:
        return
    data = await download_file(url)
    if hashlib.sha256(data).hexdigest() != checksum:
        raise ValueError(f"CLI release checksum mismatch: {url}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
