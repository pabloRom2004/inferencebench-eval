import asyncio
import json
import os
import shlex
import tempfile
import time
import uuid
from abc import abstractmethod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from http import HTTPStatus
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar, Literal, cast, overload

import anyio
import asyncssh
import httpx
import yaml
from inspect_ai.util import (
    ExecResult,
    OutputLimitExceededError,
    SandboxConnection,
    SandboxEnvironment,
    SandboxEnvironmentLimits,
    SandboxUnavailableError,
)

# Slowest link the transfer allowance assumes; larger payloads get proportionally more time.
TRANSFER_FLOOR_BYTES_PER_SECOND = 256 * 1024


def _error_detail(error: BaseException) -> str:
    """Describe a failed API call, keeping the server's explanation when it sent one."""
    if isinstance(error, httpx.HTTPStatusError):
        return f"{error.response.status_code}: {error.response.text[:500]}"
    return repr(error)


class RunPodSandbox(SandboxEnvironment):
    """Run each sample in an owned RunPod pod, using authenticated SSH for commands and files."""

    default_config: ClassVar[Path]
    working_dir: ClassVar[str]
    environment_file: ClassVar[str]
    boot_file: ClassVar[str]
    name_prefix: ClassVar[str]
    ssh_public_key_env: ClassVar[str]
    ssh_host_key_env: ClassVar[str]

    @staticmethod
    @abstractmethod
    def _configuration(path: str | None) -> dict[str, Any]:
        """Resolve and validate the task's provider configuration before allocation."""
        raise NotImplementedError

    @abstractmethod
    def _startup_command(self) -> str:
        """Return the task's boot script, including its authenticated SSH setup."""
        raise NotImplementedError

    def __init__(self, config: dict[str, Any]) -> None:
        """Generate per-pod SSH keys and defer resource creation to sample initialization."""
        super().__init__()
        # Inspect already records tools and results; SSH handshake chatter obscures them.
        asyncssh.set_log_level("WARNING")
        self.config = config
        self.pod_id: str | None = None
        self.host: str | None = None
        self.port: int | None = None
        self.key = asyncssh.generate_private_key("ssh-ed25519")
        self.host_key = asyncssh.generate_private_key("ssh-ed25519")
        self.name = self.name_prefix + uuid.uuid4().hex
        self.folder = Path("run-artifacts/runpod") / self.name
        self.ssh_folder: tempfile.TemporaryDirectory[str] | None = None

    @property
    def resource_id(self) -> str | None:
        """Identify the owned pod without exposing any credential."""
        return self.pod_id

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        """Call the RunPod REST API from the host without retrying resource-creation requests."""
        key = os.environ["RUNPOD_API_KEY"]
        attempts = (
            self.config["api_retry_attempts"] if method in {"GET", "DELETE"} else 1
        )
        async with httpx.AsyncClient(
            timeout=self.config["api_timeout_seconds"]
        ) as client:
            for attempt in range(attempts):
                try:
                    response = await client.request(
                        method,
                        self.config["api_url"] + path,
                        headers={"Authorization": f"Bearer {key}"},
                        **kwargs,
                    )
                    if (
                        method == "DELETE"
                        and response.status_code == HTTPStatus.NOT_FOUND
                    ):
                        return None
                    response.raise_for_status()
                    return response.json() if response.content else None
                except (httpx.TransportError, httpx.HTTPStatusError) as error:
                    if (
                        isinstance(error, httpx.HTTPStatusError)
                        and error.response.status_code != HTTPStatus.TOO_MANY_REQUESTS
                        and error.response.status_code
                        < HTTPStatus.INTERNAL_SERVER_ERROR
                    ):
                        raise
                    if attempt + 1 == attempts:
                        raise
                    await asyncio.sleep(self.config["poll_interval_seconds"])

    def _record(self, status: str, **details: Any) -> None:
        """Keep resource IDs and failures recoverable after a host interruption without storing keys."""
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self.folder / "pod.json"
        record = json.loads(path.read_text()) if path.exists() else {}
        record.update(name=self.name, pod_id=self.pod_id, status=status, **details)
        path.write_text(json.dumps(record, indent=2))

    @asynccontextmanager
    async def _connect(self) -> AsyncIterator[asyncssh.SSHClientConnection]:
        """Retry connection failures before sending any command; never replay remote work."""
        for attempt in range(self.config["api_retry_attempts"]):
            try:
                connection = await asyncssh.connect(
                    cast(str, self.host),
                    port=cast(int, self.port),
                    username="root",
                    client_keys=[self.key],
                    agent_path=None,
                    known_hosts=([self.host_key.convert_to_public()], [], []),
                    connect_timeout=self.config["api_timeout_seconds"],
                    keepalive_interval=self.config["ssh_keepalive_interval_seconds"],
                    keepalive_count_max=self.config["ssh_keepalive_count_max"],
                )
                break
            except (OSError, asyncssh.ConnectionLost) as error:
                if attempt + 1 == self.config["api_retry_attempts"]:
                    raise SandboxUnavailableError(
                        f"RunPod SSH connection failed for {self.pod_id}: {type(error).__name__}: {error}"
                    ) from error
                await asyncio.sleep(self.config["poll_interval_seconds"])
        async with connection:
            yield connection

    async def connection(self, *, user: str | None = None) -> SandboxConnection:
        """Expose a pinned SSH command for live debugging, keeping its private key outside the repository."""
        if not self.host or not self.port:
            raise ConnectionError("RunPod SSH is not ready")
        if self.ssh_folder is None or not Path(self.ssh_folder.name).is_dir():
            self.ssh_folder = tempfile.TemporaryDirectory(prefix="inspect-runpod-ssh-")
        folder = Path(self.ssh_folder.name)
        key = folder / "id_ed25519"
        key.write_bytes(self.key.export_private_key())
        key.chmod(0o600)
        known_hosts = folder / "known_hosts"
        known_hosts.write_text(
            f"[{self.host}]:{self.port} " + self.host_key.export_public_key().decode()
        )
        return SandboxConnection(
            type="ssh",
            command=shlex.join(
                [
                    "ssh",
                    "-i",
                    str(key),
                    "-p",
                    str(self.port),
                    "-o",
                    "IdentitiesOnly=yes",
                    "-o",
                    "StrictHostKeyChecking=yes",
                    "-o",
                    f"ServerAliveInterval={self.config['ssh_keepalive_interval_seconds']}",
                    "-o",
                    f"ServerAliveCountMax={self.config['ssh_keepalive_count_max']}",
                    "-o",
                    f"UserKnownHostsFile={known_hosts}",
                    f"{user or 'root'}@{self.host}",
                ]
            ),
        )

    async def _wait_ready(self, previous_boot: str | None) -> None:
        """Wait for authenticated SSH and, after restart, a newly generated boot marker."""
        deadline = time.monotonic() + self.config["startup_timeout_seconds"]
        while time.monotonic() < deadline:
            pod = await self._request("GET", f"/pods/{self.pod_id}")
            self.host = pod.get("publicIp")
            self.port = (pod.get("portMappings") or {}).get("22")
            if self.host and self.port:
                try:
                    result = await self.exec(
                        ["cat", self.boot_file],
                        timeout=self.config["api_timeout_seconds"],
                    )
                    if (
                        result.success
                        and result.stdout.strip()
                        and result.stdout.strip() != previous_boot
                    ):
                        connection = await self.connection()
                        self._record("ready", ssh_command=connection.command)
                        return
                except (SandboxUnavailableError, TimeoutError):
                    pass
            await asyncio.sleep(self.config["poll_interval_seconds"])
        raise TimeoutError(f"RunPod {self.pod_id} did not become ready")

    @classmethod
    async def task_init(cls, task_name: str, config: Any) -> None:
        """Validate the image and credentials before allocating a billable pod."""
        cls._configuration(config)

    @classmethod
    async def sample_init(
        cls, task_name: str, config: Any, metadata: dict[str, Any]
    ) -> dict[str, SandboxEnvironment]:
        """Create one pod per sample and release it if provisioning or SSH setup fails."""
        env = cls(cls._configuration(config))
        env._record("creating")
        payload = {
            **env.config["pod"],
            "name": env.name,
            "dockerEntrypoint": ["bash", "-c"],
            "dockerStartCmd": [env._startup_command()],
            "env": {
                env.ssh_public_key_env: env.key.export_public_key().decode(),
                env.ssh_host_key_env: env.host_key.export_private_key().decode(),
            },
        }
        try:
            pod = await env._create_pod(payload)
            env.pod_id = pod["id"]
            env._record("allocated")
            await env._wait_ready(None)
            return {"default": env}
        except BaseException as error:
            # Artifact failures must not prevent cleanup of a billable allocation.
            with suppress(OSError, ValueError):
                env._record("provision_failed", provision_error=_error_detail(error))
            with anyio.CancelScope(shield=True):
                # Recover an accepted create request whose response was lost.
                if env.pod_id is None:
                    for attempt in range(env.config["api_retry_attempts"]):
                        pods = await env._request("GET", "/pods")
                        for pod in pods:
                            if pod["name"] == env.name:
                                env.pod_id = pod["id"]
                                await env.terminate()
                        if env.pod_id is not None:
                            break
                        if attempt + 1 < env.config["api_retry_attempts"]:
                            await asyncio.sleep(env.config["poll_interval_seconds"])
                    if env.pod_id is None:
                        env._record("allocation_unconfirmed")
                else:
                    await env.terminate()
            raise

    async def _create_pod(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Request a pod, waiting out capacity and server failures that RunPod reports as 5xx."""
        attempts = self.config["create_retry_attempts"]
        for attempt in range(1, attempts + 1):
            try:
                return cast(
                    dict[str, Any], await self._request("POST", "/pods", json=payload)
                )
            except httpx.HTTPStatusError as error:
                status = error.response.status_code
                if (
                    status != HTTPStatus.TOO_MANY_REQUESTS
                    and status < HTTPStatus.INTERNAL_SERVER_ERROR
                ) or attempt == attempts:
                    raise
                # A rejected request must never be duplicated if it allocated after all.
                for pod in await self._request("GET", "/pods"):
                    if pod["name"] == self.name:
                        return cast(dict[str, Any], pod)
                self._record(
                    "create_retry", attempt=attempt, create_error=_error_detail(error)
                )
                await asyncio.sleep(self.config["create_retry_interval_seconds"])

        raise RuntimeError("RunPod create attempts exhausted")

    @classmethod
    async def sample_cleanup(
        cls,
        task_name: str,
        config: Any,
        environments: dict[str, SandboxEnvironment],
        interrupted: bool,
    ) -> None:
        """Delete owned pods, including their attached volumes, on completion or interruption."""
        with anyio.CancelScope(shield=True):
            for env in environments.values():
                await cast(RunPodSandbox, env).terminate()

    @classmethod
    async def cli_cleanup(cls, id: str | None) -> None:
        """Delete one explicitly identified pod after an interrupted host process."""
        if not id:
            raise ValueError(
                "Supply the pod ID recorded in run-artifacts/runpod; bulk deletion is not supported"
            )
        env = cls(yaml.safe_load(cls.default_config.read_text()))
        env.pod_id = id
        await env.terminate()

    async def terminate(self) -> None:
        """Release the owned pod and its volume, treating an already deleted pod as cleaned up."""
        if self.pod_id:
            try:
                await self._request("DELETE", f"/pods/{self.pod_id}")
            except Exception as error:
                self._record("cleanup_failed", cleanup_error=repr(error))
                raise
            try:
                self._record("terminated")
            finally:
                # Later connection queries must report the pod gone rather than touch removed files.
                if self.ssh_folder is not None:
                    self.ssh_folder.cleanup()
                self.ssh_folder = None
                self.host = None
                self.port = None

    async def exec(
        self,
        cmd: list[str],
        input: str | bytes | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | None = None,
        timeout: int | None = None,
        timeout_retry: bool = True,
        concurrency: bool = True,
    ) -> ExecResult[str]:
        """Execute quoted commands with bounded output and a remote process-group timeout."""
        command = list(cmd)
        if timeout is not None:
            command = ["timeout", "--kill-after=10", str(timeout), *command]
        script = (
            "source "
            + shlex.quote(self.environment_file)
            + "; cd "
            + shlex.quote(str(PurePosixPath(self.working_dir) / (cwd or ".")))
        )
        script += " && exec " + shlex.join(
            ["env", *[f"{key}={value}" for key, value in (env or {}).items()], *command]
        )
        command = ["bash", "-c", script]
        if user not in (None, "root", "0"):
            command = ["runuser", "-u", cast(str, user), "--", *command]

        # Background descendants can retain SSH pipes after the command exits.
        marker = "\nINSPECT_EXIT_" + uuid.uuid4().hex + ":"
        script = shlex.join(command) + "; inspect_status=$?; "
        # Coreutils buffers each record into one pipe write; bash printf can interleave
        # its marker, status, and newline with a noisy background descendant.
        for destination in ["", " >&2"]:
            script += (
                "/usr/bin/printf '%s%d\\n' "
                + shlex.quote(marker)
                + ' "$inspect_status"'
                + destination
                + "; "
            )
        command = ["bash", "-c", script + 'exit "$inspect_status"']
        drainers = []

        async def discard(stream: asyncssh.SSHReader[bytes]) -> None:
            """Keep channel flow control moving after this stream's completion marker."""
            while await stream.read(65536):
                pass

        async def read_tail(stream: asyncssh.SSHReader[bytes]) -> tuple[str, int]:
            """Drain through the command's completion record without waiting for descendant-held pipes."""
            data = bytearray()
            while chunk := await stream.read(65536):
                data.extend(chunk)
                before, found, after = data.partition(marker.encode())
                if found and b"\n" in after:
                    drainers.append(asyncio.create_task(discard(stream)))
                    return (
                        before[-SandboxEnvironmentLimits.MAX_EXEC_OUTPUT_SIZE :].decode(
                            errors="replace"
                        ),
                        int(after.split(b"\n", 1)[0]),
                    )
                del data[
                    : -(SandboxEnvironmentLimits.MAX_EXEC_OUTPUT_SIZE + len(marker) + 4)
                ]
            raise SandboxUnavailableError(
                "RunPod SSH closed without a command completion record"
            )

        try:
            async with self._connect() as connection:
                async with connection.create_process(
                    shlex.join(command), encoding=None
                ) as process:
                    if input is not None:
                        process.stdin.write(
                            input.encode() if isinstance(input, str) else input
                        )
                    process.stdin.write_eof()
                    async with asyncio.timeout(
                        timeout + self.config["api_timeout_seconds"]
                        if timeout is not None
                        else None
                    ):
                        stdout, stderr = await asyncio.gather(
                            read_tail(process.stdout), read_tail(process.stderr)
                        )
                        process.close()
                        await process.wait_closed()
                    returncode = stdout[1]
                    if returncode != stderr[1]:
                        raise SandboxUnavailableError(
                            "RunPod SSH returned inconsistent command completion records"
                        )
                    if timeout is not None and returncode in (124, 137):
                        raise TimeoutError("RunPod command exceeded its timeout")
                    return ExecResult(returncode == 0, returncode, stdout[0], stderr[0])
        except TimeoutError:
            raise
        except (asyncssh.Error, OSError) as error:
            raise SandboxUnavailableError(
                f"RunPod SSH failed for {self.pod_id}: {error}"
            ) from error
        finally:
            for drainer in drainers:
                drainer.cancel()
            await asyncio.gather(*drainers, return_exceptions=True)

    def transfer_allowance(self, transfer_bytes: int) -> float:
        """Allow the base API timeout plus the time a slow link needs to move the payload."""
        return (
            float(self.config["api_timeout_seconds"])
            + transfer_bytes / TRANSFER_FLOOR_BYTES_PER_SECOND
        )

    @asynccontextmanager
    async def _sftp(
        self, transfer_bytes: int = 0
    ) -> AsyncIterator[asyncssh.SFTPClient]:
        """Open an authenticated file channel with a time allowance scaled to the transfer size."""
        allowance = self.transfer_allowance(transfer_bytes)
        try:
            async with self._connect() as connection:
                async with asyncio.timeout(allowance):
                    async with connection.start_sftp_client() as sftp:
                        yield sftp
        except asyncssh.SFTPNoSuchFile as error:
            raise FileNotFoundError(str(error)) from error
        except asyncssh.SFTPPermissionDenied as error:
            raise PermissionError(str(error)) from error
        except RuntimeError as error:
            # asyncssh reports a transfer cut off by the timeout as an empty RuntimeError.
            if str(error):
                raise
            raise TimeoutError(
                f"RunPod SFTP transfer exceeded its {allowance:.0f}s allowance"
            ) from error

    async def write_file(self, file: str, contents: str | bytes) -> None:
        """Write binary or UTF-8 content through SFTP, creating its parent directories."""
        path = str(PurePosixPath(self.working_dir) / file)
        data = contents.encode() if isinstance(contents, str) else contents
        async with self._sftp(len(data)) as sftp:
            await sftp.makedirs(str(PurePosixPath(path).parent), exist_ok=True)
            async with sftp.open(path, "wb") as remote:
                await remote.write(data)

    @overload
    async def read_file(self, file: str, text: Literal[True] = True) -> str: ...

    @overload
    async def read_file(self, file: str, text: Literal[False]) -> bytes: ...

    async def read_file(self, file: str, text: bool = True) -> str | bytes:
        """Read a file with Inspect's size limit, raising rather than returning truncated data."""
        path = str(PurePosixPath(self.working_dir) / file)
        async with self._sftp(SandboxEnvironmentLimits.MAX_READ_FILE_SIZE) as sftp:
            if (await sftp.stat(path)).type == asyncssh.FILEXFER_TYPE_DIRECTORY:
                raise IsADirectoryError(path)
            async with sftp.open(path, "rb") as remote:
                data = cast(
                    bytes,
                    await remote.read(SandboxEnvironmentLimits.MAX_READ_FILE_SIZE + 1),
                )
        if len(data) > SandboxEnvironmentLimits.MAX_READ_FILE_SIZE:
            raise OutputLimitExceededError(
                SandboxEnvironmentLimits.MAX_READ_FILE_SIZE_STR, None
            )
        return data.decode() if text else data

    async def download(self, remote: str, local: str) -> None:
        """Transfer trusted evaluator artifacts without embedding binary contents in tool output."""
        async with self._sftp(4 * 1024**3) as sftp:
            await sftp.get(remote, local)

    async def upload(self, local: str, remote: str) -> None:
        """Restore a host artifact into the restarted pod through SFTP."""
        async with self._sftp(Path(local).stat().st_size) as sftp:
            await sftp.put(local, remote)
