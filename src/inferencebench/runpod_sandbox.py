import asyncio
import json
import os
import shlex
import tempfile
import time
import uuid
from contextlib import asynccontextmanager, suppress
from pathlib import Path, PurePosixPath

import anyio
import asyncssh
import httpx
from inspect_ai.util import (
    ExecResult,
    OutputLimitExceededError,
    SandboxConnection,
    SandboxEnvironment,
    SandboxEnvironmentLimits,
    SandboxUnavailableError,
    sandboxenv,
)

from inferencebench.prompts import ASSETS
from inferencebench.run_config import load_config


@sandboxenv(name="inferencebench_runpod")
class RunPodSandbox(SandboxEnvironment):
    """Run each sample in an owned RunPod pod, using authenticated SSH for commands and files."""

    def __init__(self, config: dict):
        """Generate per-pod SSH keys and defer resource creation to sample initialization."""
        super().__init__()
        self.config = config
        self.pod_id = None
        self.host = None
        self.port = None
        self.key = asyncssh.generate_private_key("ssh-ed25519")
        self.host_key = asyncssh.generate_private_key("ssh-ed25519")
        self.name = "inferencebench-" + uuid.uuid4().hex
        self.folder = Path("run-artifacts/runpod") / self.name
        self.ssh_folder = None

    @property
    def resource_id(self) -> str:
        """Identify the owned pod without exposing any credential."""
        return self.pod_id

    async def _request(self, method: str, path: str, **kwargs):
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
                    if method == "DELETE" and response.status_code == 404:
                        return None
                    response.raise_for_status()
                    return response.json() if response.content else None
                except (httpx.TransportError, httpx.HTTPStatusError) as error:
                    if (
                        isinstance(error, httpx.HTTPStatusError)
                        and error.response.status_code != 429
                        and error.response.status_code < 500
                    ):
                        raise
                    if attempt + 1 == attempts:
                        raise
                    await asyncio.sleep(self.config["poll_interval_seconds"])

    def _record(self, status: str, **details) -> None:
        """Keep resource IDs and failures recoverable after a host interruption without storing keys."""
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self.folder / "pod.json"
        record = json.loads(path.read_text()) if path.exists() else {}
        record.update(name=self.name, pod_id=self.pod_id, status=status, **details)
        path.write_text(json.dumps(record, indent=2))

    def _connect(self):
        """Authenticate with ephemeral client keys and verify the host key installed at pod creation."""
        return asyncssh.connect(
            self.host,
            port=self.port,
            username="root",
            client_keys=[self.key],
            agent_path=None,
            known_hosts=([self.host_key.convert_to_public()], [], []),
            connect_timeout=self.config["api_timeout_seconds"],
            keepalive_interval=self.config["ssh_keepalive_interval_seconds"],
            keepalive_count_max=self.config["ssh_keepalive_count_max"],
        )

    async def connection(self, *, user: str | None = None) -> SandboxConnection:
        """Expose a pinned SSH command for live debugging, keeping its private key outside the repository."""
        if not self.host or not self.port:
            raise ConnectionError("RunPod SSH is not ready")
        if self.ssh_folder is None:
            self.ssh_folder = tempfile.TemporaryDirectory(prefix="inferencebench-ssh-")
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
                        ["cat", "/run/inferencebench-boot-id"],
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
    async def task_init(cls, task_name, config) -> None:
        """Validate the image and credentials before allocating a billable pod."""
        cls._configuration(config)

    def _startup_command(self) -> str:
        """Install the Dockerfile's shared environment on first boot, or restore it after restart."""
        dockerfile = (ASSETS.parent / "Dockerfile").read_text().replace("\\\n", "")
        exports = [
            "export " + line[4:]
            for line in dockerfile.splitlines()
            if line.startswith("ENV ")
        ]
        script = "set -e\n" + "\n".join(exports) + "\n"
        if self.config["bootstrap"]:
            script += "if [ ! -f /workspace/.inferencebench/ready ]; then\n"
            script += (
                "bash -c "
                + shlex.quote((ASSETS / "scripts" / "setup_environment.sh").read_text())
                + "\n"
            )
            script += "elif ! command -v rsync >/dev/null || ! command -v sshd >/dev/null; then\n"
            script += "apt-get update && apt-get install -y rsync openssh-server python3\nfi\n"
        script += "mkdir -p /opt\nprintf %s " + shlex.quote(
            (ASSETS / "scripts" / "runpod_start.sh").read_text()
        )
        script += " > /opt/runpod_start.sh\nexec bash /opt/runpod_start.sh"
        return script

    @staticmethod
    def _configuration(path) -> dict:
        """Resolve the complete provider YAML and reject missing prerequisites before a paid API call."""
        config = load_config(path or "runpod.yaml")
        # Existing complete provider configs predate the transport keepalive controls.
        defaults = load_config("runpod.yaml")
        for name in ["ssh_keepalive_interval_seconds", "ssh_keepalive_count_max"]:
            config.setdefault(name, defaults[name])
        config["pod"]["imageName"] = (
            os.environ.get("RUNPOD_IMAGE") or config["pod"]["imageName"]
        )
        if not os.environ.get("RUNPOD_API_KEY") or not config["pod"]["imageName"]:
            raise ValueError(
                "RunPod requires RUNPOD_API_KEY and an image in RUNPOD_IMAGE or pod.imageName"
            )
        if type(config["bootstrap"]) is not bool:
            raise ValueError("bootstrap must be true or false")
        if (
            config["pod"]["volumeMountPath"] != "/workspace"
            or config["pod"]["volumeInGb"] <= 0
        ):
            raise ValueError(
                "RunPod requires a persistent volume at /workspace for the scoring restart"
            )
        for name in [
            "api_timeout_seconds",
            "api_retry_attempts",
            "ssh_keepalive_interval_seconds",
            "ssh_keepalive_count_max",
            "startup_timeout_seconds",
            "snapshot_timeout_seconds",
            "poll_interval_seconds",
        ]:
            if type(config[name]) is not int or config[name] <= 0:
                raise ValueError(f"{name} must be a positive integer")
        return config

    @classmethod
    async def sample_init(cls, task_name, config, metadata):
        """Create one pod per sample and release it if provisioning or SSH setup fails."""
        env = cls(cls._configuration(config))
        env._record("creating")
        payload = {
            **env.config["pod"],
            "name": env.name,
            "dockerEntrypoint": ["bash", "-c"],
            "dockerStartCmd": [env._startup_command()],
            "env": {
                "INFERENCEBENCH_SSH_PUBLIC_KEY": env.key.export_public_key().decode(),
                "INFERENCEBENCH_SSH_HOST_KEY": env.host_key.export_private_key().decode(),
            },
        }
        try:
            pod = await env._request("POST", "/pods", json=payload)
            env.pod_id = pod["id"]
            env._record("allocated")
            await env._wait_ready(None)
            return {"default": env}
        except BaseException as error:
            # Artifact failures must not prevent cleanup of a billable allocation.
            with suppress(OSError, ValueError):
                env._record("provision_failed", provision_error=repr(error))
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

    @classmethod
    async def sample_cleanup(cls, task_name, config, environments, interrupted) -> None:
        """Delete owned pods, including their attached volumes, on completion or interruption."""
        with anyio.CancelScope(shield=True):
            for env in environments.values():
                await env.terminate()

    @classmethod
    async def cli_cleanup(cls, id: str | None) -> None:
        """Delete one explicitly identified pod after an interrupted host process."""
        if not id:
            raise ValueError(
                "Supply the pod ID recorded in run-artifacts/runpod; bulk deletion is not supported"
            )
        env = cls(load_config("runpod.yaml"))
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
                if self.ssh_folder is not None:
                    self.ssh_folder.cleanup()

    async def restart(self, config_file: str | None) -> "RunPodSandbox":
        """Snapshot the installed filesystem to the volume and reboot the pod before grading."""
        boot = await self.read_file("/run/inferencebench-boot-id")
        await self.write_file(
            "/opt/runpod_start.sh", (ASSETS / "scripts" / "runpod_start.sh").read_text()
        )
        result = await self.exec(
            ["bash", "/opt/runpod_start.sh", "snapshot"],
            timeout=self.config["snapshot_timeout_seconds"],
        )
        if not result.success:
            raise RuntimeError(f"RunPod filesystem snapshot failed: {result.stderr}")
        for attempt in range(self.config["api_retry_attempts"]):
            self._record("restarting", restart_attempt=attempt + 1)
            try:
                await self._request("POST", f"/pods/{self.pod_id}/restart")
            except (httpx.TransportError, httpx.HTTPStatusError) as error:
                if (
                    isinstance(error, httpx.HTTPStatusError)
                    and error.response.status_code != 429
                    and error.response.status_code < 500
                ):
                    raise
                self._record("restart_response_failed", restart_error=repr(error))
                # A failed response may follow an accepted restart. Allow the full
                # restoration window before considering another restart request.
                try:
                    await self._wait_ready(boot.strip())
                    return self
                except TimeoutError:
                    if attempt + 1 == self.config["api_retry_attempts"]:
                        raise error
            else:
                await self._wait_ready(boot.strip())
                return self
        return self

    async def exec(
        self,
        cmd,
        input=None,
        cwd=None,
        env=None,
        user=None,
        timeout=None,
        timeout_retry=True,
        concurrency=True,
    ) -> ExecResult[str]:
        """Execute quoted commands with bounded output and a remote process-group timeout."""
        command = list(cmd)
        if timeout is not None:
            command = ["timeout", "--kill-after=10", str(timeout), *command]
        script = "source /etc/inferencebench-env.sh; cd " + shlex.quote(
            str(PurePosixPath("/home/agent/task") / (cwd or "."))
        )
        script += " && exec " + shlex.join(
            ["env", *[f"{key}={value}" for key, value in (env or {}).items()], *command]
        )
        command = ["bash", "-c", script]
        if user not in (None, "root", "0"):
            command = ["runuser", "-u", user, "--", *command]

        # Background descendants can retain SSH pipes after the command exits.
        marker = "\nINFERENCEBENCH_EXIT_" + uuid.uuid4().hex + ":"
        script = shlex.join(command) + "; inferencebench_status=$?; "
        # Coreutils buffers each record into one pipe write; bash printf can interleave
        # its marker, status, and newline with a noisy background descendant.
        for destination in ["", " >&2"]:
            script += (
                "/usr/bin/printf '%s%d\\n' " + shlex.quote(marker)
                + ' "$inferencebench_status"' + destination + "; "
            )
        command = ["bash", "-c", script + 'exit "$inferencebench_status"']
        drainers = []

        async def discard(stream):
            """Keep channel flow control moving after this stream's completion marker."""
            while await stream.read(65536):
                pass

        async def read_tail(stream):
            """Drain through the command's completion record without waiting for descendant-held pipes."""
            data = bytearray()
            while chunk := await stream.read(65536):
                data.extend(chunk)
                before, found, after = data.partition(marker.encode())
                if found and b"\n" in after:
                    drainers.append(asyncio.create_task(discard(stream)))
                    return (
                        before[-SandboxEnvironmentLimits.MAX_EXEC_OUTPUT_SIZE:].decode(errors="replace"),
                        int(after.split(b"\n", 1)[0]),
                    )
                del data[: -(SandboxEnvironmentLimits.MAX_EXEC_OUTPUT_SIZE + len(marker) + 4)]
            raise SandboxUnavailableError("RunPod SSH closed without a command completion record")

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
                        raise SandboxUnavailableError("RunPod SSH returned inconsistent command completion records")
                    if timeout is not None and returncode in (124, 137):
                        raise TimeoutError("RunPod command exceeded its timeout")
                    return ExecResult(
                        returncode == 0, returncode, stdout[0], stderr[0]
                    )
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

    @asynccontextmanager
    async def _sftp(self):
        """Open an authenticated file channel and preserve Inspect's file-error semantics."""
        try:
            async with asyncio.timeout(self.config["api_timeout_seconds"]):
                async with (
                    self._connect() as connection,
                    connection.start_sftp_client() as sftp,
                ):
                    yield sftp
        except asyncssh.SFTPNoSuchFile as error:
            raise FileNotFoundError(str(error)) from error
        except asyncssh.SFTPPermissionDenied as error:
            raise PermissionError(str(error)) from error

    async def write_file(self, file: str, contents: str | bytes) -> None:
        """Write binary or UTF-8 content through SFTP, creating its parent directories."""
        path = str(PurePosixPath("/home/agent/task") / file)
        async with self._sftp() as sftp:
            await sftp.makedirs(str(PurePosixPath(path).parent), exist_ok=True)
            async with sftp.open(path, "wb") as remote:
                await remote.write(
                    contents.encode() if isinstance(contents, str) else contents
                )

    async def read_file(self, file: str, text: bool = True) -> str | bytes:
        """Read a file with Inspect's size limit, raising rather than returning truncated data."""
        path = str(PurePosixPath("/home/agent/task") / file)
        async with self._sftp() as sftp:
            if (await sftp.stat(path)).type == asyncssh.FILEXFER_TYPE_DIRECTORY:
                raise IsADirectoryError(path)
            async with sftp.open(path, "rb") as remote:
                data = await remote.read(
                    SandboxEnvironmentLimits.MAX_READ_FILE_SIZE + 1
                )
        if len(data) > SandboxEnvironmentLimits.MAX_READ_FILE_SIZE:
            raise OutputLimitExceededError(
                SandboxEnvironmentLimits.MAX_READ_FILE_SIZE_STR, None
            )
        return data.decode() if text else data

    async def download(self, remote: str, local: str) -> None:
        """Transfer trusted evaluator artifacts without embedding binary contents in tool output."""
        async with self._sftp() as sftp:
            await sftp.get(remote, local)

    async def upload(self, local: str, remote: str) -> None:
        """Restore a host artifact into the restarted pod through SFTP."""
        async with self._sftp() as sftp:
            await sftp.put(local, remote)
