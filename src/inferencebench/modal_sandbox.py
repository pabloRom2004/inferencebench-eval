from pathlib import PurePosixPath

import modal
from inspect_ai.util import sandboxenv
from inspect_sandboxes.modal._modal import ModalSandboxEnvironment

from inferencebench.run_config import load_config


@sandboxenv(name="inferencebench_modal")
class InferenceSandbox(ModalSandboxEnvironment):
    """Use Modal's current filesystem API until inspect-sandboxes migrates its retired FileIO calls."""

    @property
    def resource_id(self) -> str:
        """Identify the currently allocated Modal sandbox in evaluation artifacts."""
        return self.sandbox.object_id

    async def download(self, remote: str, local: str) -> None:
        """Copy a scoring artifact from Modal to the host without a tool-output limit."""
        await self.sandbox.filesystem.copy_to_local.aio(remote, local)

    async def upload(self, local: str, remote: str) -> None:
        """Restore a trusted host artifact into the scoring sandbox."""
        await self.sandbox.filesystem.copy_from_local.aio(local, remote)

    async def terminate(self) -> None:
        """Release the sandbox allocated for this attempt."""
        await self.sandbox.terminate.aio(wait=True)

    async def restart(self, config_file: str | None) -> "InferenceSandbox":
        """Preserve the submitted filesystem and allocate a fresh H100 without its running processes."""
        image = await self.sandbox.snapshot_filesystem.aio(timeout=55)
        await self.terminate()
        config = load_config(config_file or "compose.yaml")
        resources = config["services"]["default"]
        app = await modal.App.lookup.aio(
            "inferencebench-scoring", create_if_missing=True
        )
        remote = await modal.Sandbox.create.aio(
            "sleep",
            "infinity",
            app=app,
            image=image,
            gpu=config["x-modal"]["gpu"],
            cpu=resources["cpus"],
            memory=int(resources["mem_limit"].removesuffix("g")) * 1024,
            timeout=config["x-modal"]["timeout"],
            workdir=resources["working_dir"],
        )
        return InferenceSandbox(remote)

    async def _absolute_file(self, file: str) -> str:
        """Resolve relative Inspect paths against the sandbox's configured working directory."""
        if PurePosixPath(file).is_absolute():
            return file
        result = await self.exec(["pwd"])
        if not result.success:
            raise RuntimeError(result.stderr)
        return str(PurePosixPath(result.stdout.strip()) / file)

    async def _create_parent_folder(self, path: str) -> None:
        """Create parent directories through Modal's replacement filesystem service."""
        await self.sandbox.filesystem.make_directory.aio(
            await self._absolute_file(path)
        )

    async def _write_file_content(self, file: str, contents: str | bytes) -> None:
        """Write UTF-8 text or bytes without opening a legacy remote file descriptor."""
        data = contents.encode() if isinstance(contents, str) else contents
        await self.sandbox.filesystem.write_bytes.aio(
            data, await self._absolute_file(file)
        )

    async def _read_file_content(self, file: str) -> bytes:
        """Read file bytes through the supported path-oriented filesystem service."""
        return await self.sandbox.filesystem.read_bytes.aio(
            await self._absolute_file(file)
        )
