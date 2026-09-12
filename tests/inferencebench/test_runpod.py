"""Test provider lifecycle locally; the Docker fixture uses synthetic metrics and no GPU or paid API."""

import asyncio
import importlib
import json
import os
import shlex
import subprocess
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import asyncssh
import httpx
import pytest
import yaml
from inspect_ai import eval_async
from inspect_ai.agent import as_solver
from inspect_ai.log import read_eval_log
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.util import (
    ExecResult,
    OutputLimitExceededError,
    SandboxEnvironmentLimits,
)

from inferencebench import cli_agent, inference_bench, react_agent
from inferencebench.prompts import ASSETS
from inferencebench.runpod_sandbox import RunPodSandbox


@pytest.fixture
def provider(monkeypatch, tmp_path):
    """Supply dummy credentials and isolated records without contacting RunPod."""
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key-not-a-credential")
    monkeypatch.setenv("RUNPOD_IMAGE", "local-test-image")
    config = RunPodSandbox._configuration(None)
    config["poll_interval_seconds"] = 0
    env = RunPodSandbox(config)
    env.folder = tmp_path / "provider"
    env.pod_id = "test-pod"
    return env


@pytest.fixture(autouse=True)
def remove_mock_logs():
    """Remove only mock evaluation logs created by this test, keeping the log directory flat."""
    before = set(Path("logs").glob("*.eval"))
    yield
    for path in set(Path("logs").glob("*.eval")) - before:
        if read_eval_log(path, header_only=True).eval.model.startswith("mockllm/"):
            path.unlink()


def test_provider_selection():
    """Keep Modal as the default and select the complete RunPod provider configuration explicitly."""
    assert inference_bench(quality_cache=None).sandbox.type == "inferencebench_modal"
    task = inference_bench(gpu_provider="runpod", quality_cache=None)
    assert task.sandbox.type == "inferencebench_runpod"
    assert task.sandbox.config.endswith("runpod.yaml")
    assert task.dataset[0].metadata["gpu_provider"] == "runpod"
    with pytest.raises(ValueError, match="gpu_provider"):
        inference_bench(gpu_provider="unknown")


@pytest.mark.parametrize("keepalive", [False, True])
async def test_idle_ssh_connection(provider, keepalive):
    """Reproduce an idle network timeout and keep a silent command alive with SSH probes."""
    class Server(asyncssh.SSHServer):
        """Accept local test connections without external credentials."""

        def begin_auth(self, username):
            """Disable authentication only for the loopback fixture."""
            return False

    async def command(process):
        """Emulate preparation which only emits output on completion."""
        await asyncio.sleep(1.5)
        process.stdout.write("preparation complete")
        process.exit(0)

    handlers = set()

    async def proxy(reader, writer):
        """Close a connection when no application traffic crosses it for half a second."""
        task = asyncio.current_task()
        handlers.add(task)
        upstream_reader, upstream_writer = await asyncio.open_connection("127.0.0.1", server.get_port())
        last_activity = asyncio.get_running_loop().time()

        async def forward(source, destination):
            """Forward bytes and refresh the proxy's idle deadline."""
            nonlocal last_activity
            while data := await source.read(65536):
                last_activity = asyncio.get_running_loop().time()
                destination.write(data)
                await destination.drain()

        async def idle_timeout():
            """Reproduce a network device expiring an idle SSH flow."""
            while asyncio.get_running_loop().time() - last_activity < 0.5:
                await asyncio.sleep(0.05)

        pending = [asyncio.create_task(forward(reader, upstream_writer)),
                   asyncio.create_task(forward(upstream_reader, writer)),
                   asyncio.create_task(idle_timeout())]
        try:
            await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for item in pending:
                item.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            writer.close()
            upstream_writer.close()
            await asyncio.gather(writer.wait_closed(), upstream_writer.wait_closed(), return_exceptions=True)
            handlers.discard(task)

    async with asyncssh.listen("127.0.0.1", 0, server_factory=Server,
                               server_host_keys=[provider.host_key], process_factory=command) as server:
        async with await asyncio.start_server(proxy, "127.0.0.1", 0) as network:
            provider.host = "127.0.0.1"
            provider.port = network.sockets[0].getsockname()[1]
            provider.config["ssh_keepalive_interval_seconds"] = 0.1 if keepalive else 0
            try:
                async with provider._connect() as connection:
                    result = await connection.run("prepare", timeout=3)
                    if keepalive:
                        assert result.exit_status == 0
                        assert result.stdout == "preparation complete"
                    else:
                        assert result.exit_status is None
                        assert result.stdout == ""
            except asyncssh.ConnectionLost:
                assert not keepalive
            finally:
                if handlers:
                    await asyncio.gather(*handlers)


@pytest.mark.parametrize("missing", ["RUNPOD_API_KEY", "RUNPOD_IMAGE"])
def test_missing_prerequisites(provider, monkeypatch, missing, tmp_path):
    """Fail before resource allocation when the API key or published image is missing."""
    monkeypatch.delenv(missing)
    config = (
        RunPodSandbox._configuration(None)
        if missing == "RUNPOD_IMAGE"
        else provider.config
    )
    if missing == "RUNPOD_IMAGE":
        config["pod"]["imageName"] = None
    path = tmp_path / "runpod.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="RunPod requires"):
        RunPodSandbox._configuration(path)


async def test_nullable_network_and_restart_marker(provider, monkeypatch):
    """Poll through nullable network mappings and reject the old boot marker after restart."""
    request = AsyncMock(
        side_effect=[
            {"publicIp": None, "portMappings": None},
            {"publicIp": "127.0.0.1", "portMappings": {"22": 12345}},
            {"publicIp": "127.0.0.1", "portMappings": {"22": 12345}},
        ]
    )
    execute = AsyncMock(
        side_effect=[ExecResult(True, 0, "old", ""), ExecResult(True, 0, "new", "")]
    )
    monkeypatch.setattr(provider, "_request", request)
    monkeypatch.setattr(provider, "exec", execute)
    await provider._wait_ready("old")
    assert request.await_count == 3
    assert execute.await_count == 2


@pytest.mark.parametrize("outcome", ["rejected", "accepted", "exhausted", "forbidden"])
async def test_restart_reconciles_failed_response(provider, monkeypatch, outcome):
    """Recover transient restart errors without repeating an accepted restart or the snapshot."""
    status = 403 if outcome == "forbidden" else 500
    response = httpx.Response(status, request=httpx.Request("POST", "https://example.invalid/restart"))
    failure = httpx.HTTPStatusError("restart response failed", request=response.request, response=response)
    request = AsyncMock(side_effect=[failure, None] if outcome == "rejected" else failure)
    ready = AsyncMock(side_effect=[TimeoutError("old boot"), None] if outcome == "rejected"
                      else TimeoutError("old boot") if outcome == "exhausted" else None)
    snapshot = AsyncMock(return_value=ExecResult(True, 0, "", ""))
    monkeypatch.setattr(provider, "read_file", AsyncMock(return_value="old\n"))
    monkeypatch.setattr(provider, "write_file", AsyncMock())
    monkeypatch.setattr(provider, "exec", snapshot)
    monkeypatch.setattr(provider, "_request", request)
    monkeypatch.setattr(provider, "_wait_ready", ready)
    if outcome in {"forbidden", "exhausted"}:
        with pytest.raises(httpx.HTTPStatusError) as caught:
            await provider.restart(None)
        assert caught.value is failure
    else:
        assert await provider.restart(None) is provider
    expected = {"rejected": 2, "accepted": 1, "exhausted": 3, "forbidden": 1}[outcome]
    assert request.await_count == expected
    assert ready.await_count == (0 if outcome == "forbidden" else expected)
    snapshot.assert_awaited_once()
    assert all(call.args == ("old",) for call in ready.await_args_list)


@pytest.mark.parametrize(
    "method,failures,expected", [("DELETE", 1, 2), ("DELETE", 3, 3), ("POST", 1, 1)]
)
async def test_idempotent_api_retries(
    provider, monkeypatch, method, failures, expected
):
    """Retry transient deletion failures while making only one potentially billable create request."""
    calls = []
    client = httpx.AsyncClient

    def respond(request):
        """Emulate transient network errors and verify that credentials stay in the host request header."""
        calls.append(request)
        assert request.headers["Authorization"] == "Bearer test-key-not-a-credential"
        if len(calls) <= failures:
            raise httpx.ConnectError("temporary disconnect", request=request)
        return httpx.Response(204)

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: client(transport=httpx.MockTransport(respond), **kwargs),
    )
    if method == "DELETE":
        if failures == 3:
            with pytest.raises(httpx.ConnectError):
                await provider.terminate()
            record = json.loads((provider.folder / "pod.json").read_text())
            assert record["status"] == "cleanup_failed"
            assert record["pod_id"] == "test-pod"
        else:
            await provider.terminate()
            assert (
                json.loads((provider.folder / "pod.json").read_text())["status"]
                == "terminated"
            )
    else:
        with pytest.raises(httpx.ConnectError):
            await provider._request("POST", "/pods", json={"name": "test"})
    assert len(calls) == expected


@pytest.mark.parametrize("lost_response", [False, True])
async def test_failed_provisioning_cleanup(provider, monkeypatch, lost_response):
    """Recover a lost create response by its unique name and clean up failed provisioning."""
    calls = []

    class TestPod(RunPodSandbox):
        """Keep this failed allocation's records in the test directory."""

        def __init__(self, config):
            """Initialize real provider state with an isolated artifact directory."""
            super().__init__(config)
            self.folder = provider.folder

    monkeypatch.setattr(
        RunPodSandbox, "_configuration", staticmethod(lambda path: provider.config)
    )

    async def request(self, method, path, **kwargs):
        """Expose only this test's uniquely named allocation to reconciliation."""
        calls.append((method, path))
        if method == "POST":
            payload = kwargs["json"]
            assert set(payload["env"]) == {
                "INFERENCEBENCH_SSH_PUBLIC_KEY",
                "INFERENCEBENCH_SSH_HOST_KEY",
            }
            assert "RUNPOD_API_KEY" not in json.dumps(payload)
            if lost_response:
                raise httpx.ReadTimeout("create response lost")
            return {"id": "test-pod"}
        if method == "GET":
            return [
                {"name": self.name, "id": "test-pod"},
                {"name": "someone-else", "id": "other-pod"},
            ]
        return None

    monkeypatch.setattr(TestPod, "_request", request)
    monkeypatch.setattr(
        TestPod, "_wait_ready", AsyncMock(side_effect=TimeoutError("SSH unavailable"))
    )
    with pytest.raises((TimeoutError, httpx.ReadTimeout)):
        await TestPod.sample_init("test", None, {})
    assert ("DELETE", "/pods/test-pod") in calls
    assert ("DELETE", "/pods/other-pod") not in calls
    assert (
        json.loads((provider.folder / "pod.json").read_text())["status"] == "terminated"
    )


async def test_allocation_record_failure_still_deletes(provider, monkeypatch):
    """Delete an allocated pod even when a full host disk prevents all subsequent journal writes."""
    calls = []

    def record(self, status, **details):
        """Allow the preflight record and emulate disk exhaustion immediately after allocation."""
        if status != "creating":
            raise OSError("disk full")

    async def request(self, method, path, **kwargs):
        """Record allocation and deletion without making any network request."""
        calls.append((method, path))
        return {"id": "test-pod"} if method == "POST" else None

    monkeypatch.setattr(RunPodSandbox, "_record", record)
    monkeypatch.setattr(RunPodSandbox, "_request", request)
    with pytest.raises(OSError, match="disk full"):
        await RunPodSandbox.sample_init("test", None, {})
    assert calls == [("POST", "/pods"), ("DELETE", "/pods/test-pod")]


async def test_cleanup_record_failure_removes_ssh_key(provider, monkeypatch):
    """Remove temporary private keys after successful pod deletion even if its status cannot be saved."""
    provider.host = "127.0.0.1"
    provider.port = 2222
    await provider.connection()
    folder = Path(provider.ssh_folder.name)
    request = AsyncMock()
    monkeypatch.setattr(provider, "_request", request)

    def record(status, **details):
        """Simulate host disk failure after RunPod has confirmed deletion."""
        raise OSError("disk full")

    monkeypatch.setattr(provider, "_record", record)
    with pytest.raises(OSError, match="disk full"):
        await provider.terminate()
    request.assert_awaited_once_with("DELETE", "/pods/test-pod")
    assert not folder.exists()


@pytest.fixture(params=["reset", "preserve"])
async def docker_pods(monkeypatch, tmp_path, request):
    """Exercise fresh and retained container disks with the real SSH and startup code."""
    if os.environ.get("INFERENCEBENCH_TEST_DOCKER") != "1":
        pytest.skip(
            "Set INFERENCEBENCH_TEST_DOCKER=1 for the local Linux transport test"
        )
    disk_mode = request.param
    driver_profiles = tmp_path / "driver-profiles"
    driver_profiles.mkdir()
    (driver_profiles / "10-container.conf").write_text("provider-owned driver profile")
    image = "inferencebench-runpod-test:" + uuid.uuid4().hex[:12]
    context = tmp_path / "image"
    context.mkdir()
    (context / "scripts").mkdir()
    (context / "scripts" / "runpod_start.sh").write_bytes((ASSETS / "scripts" / "runpod_start.sh").read_bytes())
    (context / "scripts" / "setup_environment.sh").write_text(
        "echo setup >> /workspace/setup-calls\n"
    )
    (context.parent / "Dockerfile").write_text(
        (ASSETS.parent / "Dockerfile").read_text()
    )
    monkeypatch.setattr(
        importlib.import_module("inferencebench.runpod_sandbox"), "ASSETS", context
    )
    (context / "Dockerfile").write_text("""FROM ubuntu:22.04
RUN apt-get update && apt-get install -y openssh-server rsync python3 && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /home/agent/task /opt/evaluator/bin /opt/inferencebench/src/eval/inference/bin /opt/inference_eval/bin && touch /usr/local/lib/delete-me
RUN mkdir -p /etc/nvidia/nvidia-application-profiles-rc.d
COPY scripts/runpod_start.sh /opt/runpod_start.sh
COPY evaluator.py /opt/evaluator/bin/python
COPY runner.py /opt/inferencebench/src/eval/inference/runner.py
RUN chmod +x /opt/evaluator/bin/python && printf '#!/bin/sh\\necho "CPU-only synthetic GPU fixture"\\n' > /usr/local/bin/nvidia-smi && chmod +x /usr/local/bin/nvidia-smi
RUN echo trusted > /opt/inferencebench/src/eval/inference/bin/launch_supervised_server.sh
RUN mkdir -p /opt/inferencebench/src/eval/tasks/_shared/task_context && touch /opt/inferencebench/src/eval/tasks/_shared/task_context/start_server.sh /opt/inferencebench/src/eval/tasks/_shared/task_context/test_server.sh
RUN printf original > /usr/local/lib/same-stat
ENV HF_HOME=/opt/hf_cache TEST_ENV_MARKER=preserved
ENTRYPOINT ["bash", "/opt/runpod_start.sh"]
""")
    (context / "runner.py").write_text('''import argparse
import json
import os
from pathlib import Path


def build_parser():
    """Return the fixture's development-command parser."""
    return argparse.ArgumentParser()


def run_evaluation(task, args):
    """Verify that the generated wrapper selects the uploaded request file."""
    path = Path(os.environ['INFERENCE_BENCH_REQUESTS_FILE'])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]['ignore_eos']
    print('Prepared development request loaded')
''')
    (context / "evaluator.py").write_text('''#!/usr/bin/python3
"""Return synthetic measurements solely to test Inspect's provider and scoring plumbing."""
import json, os, socket, sys
from pathlib import Path
folder = Path('/tmp/inferencebench')
operation = sys.argv[2]
metrics = {'profiles': {'burst': {'success_count': 1, 'ttft': {'p50': 2 if operation == 'prepare' else 1}}}, 'quality_check': {'pass': True}}
if operation == 'prepare':
    options = json.loads((folder / 'options.json').read_text())
    for name in ['dev', 'heldout']:
        rows = [json.loads(line) for line in (folder / (name + '-requests.jsonl')).read_text().splitlines()]
        assert len(rows) == options['request_limit']
        assert 6554 <= rows[0]['target_input_token_count'] <= 8192
        assert rows[0]['ignore_eos']
    assert options['request_cache_provenance']['format_version'] == 1
    Path('/home/agent/task/requests.jsonl').write_text((folder / 'dev-requests.jsonl').read_text())
    import importlib.util
    spec = importlib.util.spec_from_file_location('runtime', folder / 'runtime.py')
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    runtime.install_workspace(options)
    from dataclasses import dataclass
    from types import SimpleNamespace
    import inference

    @dataclass
    class QualitySpec:
        """Represent the upstream quality input contract in the CPU fixture."""
        samples_file: Path
        seed: int
        limit: int

    def reference(selection, url, model, timeout, concurrency, out):
        """Force one initial timeout and require an isolated retry of that exact saved input."""
        samples = [json.loads(line) for line in selection.samples_file.read_text().splitlines()]
        retry = selection.limit == 1
        assert timeout == options['request_timeout_seconds']
        assert concurrency == (1 if retry else options['quality_concurrency'])
        if retry:
            assert samples == [{'sample_id': '0', 'max_new_tokens': 2048, 'temperature': 0}]
        rows = [{'sample_id': item['sample_id'], 'request_index': index,
                 'success': retry or index != 0, 'gold_answer': 'A',
                 'parsed_answer': 'A' if retry or index != 0 else None}
                for index, item in enumerate(samples)]
        path = out / 'baseline_generations.jsonl'
        path.write_text(''.join(json.dumps(row) + '\\n' for row in rows))
        return sum(row['success'] for row in rows) / len(rows), path, None

    def accuracy(rows):
        """Calculate a known reference accuracy after the failed request recovers."""
        return sum(row['parsed_answer'] == row['gold_answer'] for row in rows) / len(rows)

    samples_file = folder / 'quality-samples.jsonl'
    samples_file.write_text(''.join(json.dumps({'sample_id': str(index), 'max_new_tokens': 2048, 'temperature': 0}) + '\\n' for index in range(options['quality_samples'])))
    inference.precompute_quality_baseline = SimpleNamespace(_run_dataset=reference, _accuracy=accuracy)
    runtime.prepare_quality_baseline(options, QualitySpec(samples_file, options['quality_seed'], options['quality_samples']))
    assert (folder / 'quality-baseline/attempt-2/0/baseline_generations.jsonl').exists()
    (folder / 'baseline.json').write_text(json.dumps(metrics))
    (folder / 'provenance.json').write_text(json.dumps({'downloaded_model_revision': 'cpu-fixture', 'input_sha256': {}}))
else:
    if os.environ['TEST_DISK_MODE'] == 'reset':
        assert Path('/etc/nvidia/nvidia-application-profiles-rc.d/10-container.conf').read_text() == 'provider-owned driver profile'
    assert Path('/workspace/setup-calls').read_text() == 'setup\\n'
    assert Path('/usr/local/lib/agent-change').read_text() == 'preserved'
    original = Path('/usr/local/lib/agent-change')
    assert original.stat().st_mtime_ns == 1700000000123456789
    assert original.stat().st_ino == Path('/usr/local/lib/agent-hardlink').stat().st_ino
    assert os.readlink('/usr/local/lib/agent-symlink') == 'agent-change'
    sparse = Path('/usr/local/lib/agent-sparse').stat()
    assert sparse.st_size == 1073741824 and sparse.st_blocks * 512 < 1048576
    assert Path('/usr/local/lib/same-stat').read_text() == 'modified'
    assert not Path('/usr/local/lib/delete-me').exists()
    assert Path('/run/agent-ephemeral').exists() == (os.environ['TEST_DISK_MODE'] == 'preserve')
    assert Path('/tmp/live-writer.log').stat().st_size > 0
    assert Path('/opt/inferencebench/src/eval/inference/bin/launch_supervised_server.sh').read_text().strip() == 'trusted'
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 8765))
    (folder / 'final.json').write_text(json.dumps(metrics))
    (folder / 'final-server.log').write_text('CPU fixture: process reset and filesystem persistence passed')
''')
    build = await asyncio.to_thread(
        subprocess.run,
        ["docker", "build", "-t", image, str(context)],
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    pods = {}
    calls = []

    async def docker(*args):
        """Operate only containers and volumes created by this fixture."""
        process = await asyncio.create_subprocess_exec(
            "docker",
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode:
            raise RuntimeError(stderr.decode())
        return stdout.decode().strip()

    async def start(pod):
        """Start a pristine container with the preserved volume and per-pod SSH keys."""
        args = ["run", "-d", "-p", "127.0.0.1::22", "-v", pod["volume"] + ":/workspace",
                "--tmpfs", "/inferencebench-capacity:size=32m"]
        if pod.get("restarted"):
            args += ["--mount", f"type=bind,source={driver_profiles},target=/etc/nvidia/nvidia-application-profiles-rc.d,readonly"]
        for name, value in pod["env"].items():
            args += ["-e", name + "=" + value]
        pod["container"] = await docker(
            *args, "--entrypoint", "bash", image, "-c", pod["command"]
        )

    async def request(self, method, path, **kwargs):
        """Map RunPod REST operations onto local containers without contacting a cloud service."""
        calls.append((method, path))
        if method == "POST" and path == "/pods":
            body = kwargs["json"]
            pod_id = uuid.uuid4().hex
            pod = {
                "id": pod_id,
                "name": body["name"],
                "env": {**body["env"], "TEST_DISK_MODE": disk_mode},
                "volume": "inferencebench-test-" + pod_id,
                "command": body["dockerStartCmd"][0],
            }
            pods[pod_id] = pod
            await docker("volume", "create", pod["volume"])
            await start(pod)
            return {"id": pod_id}
        pod_id = path.split("/")[2]
        pod = pods.get(pod_id)
        if method == "DELETE":
            if pod:
                await docker("rm", "-f", pod["container"])
                await docker("volume", "rm", pod["volume"])
                del pods[pod_id]
            return None
        if method == "POST":
            if disk_mode == "preserve":
                await docker("restart", "-t", "1", pod["container"])
            else:
                await docker("rm", "-f", pod["container"])
                pod["restarted"] = True
                await start(pod)
            # Exercise an accepted restart whose response fails through the real
            # SSH restore and Inspect scoring path, in both disk lifecycle modes.
            raise httpx.ReadTimeout("restart response lost after container restart")
        record = json.loads(await docker("inspect", pod["container"]))[0]
        if not record["State"]["Running"]:
            raise RuntimeError(str(record["State"]) + "\n" + await docker("logs", pod["container"]))
        port = record["NetworkSettings"]["Ports"]["22/tcp"][0]["HostPort"]
        return {"publicIp": "127.0.0.1", "portMappings": {"22": int(port)}}

    monkeypatch.setenv("RUNPOD_API_KEY", "test-key-not-a-credential")
    monkeypatch.setenv("RUNPOD_IMAGE", image)
    monkeypatch.setattr(RunPodSandbox, "_request", request)
    try:
        yield SimpleNamespace(pods=pods, calls=calls)
    finally:
        for pod in list(pods.values()):
            await docker("rm", "-f", pod["container"])
            await docker("volume", "rm", pod["volume"])
        await docker("image", "rm", image)


@pytest.mark.parametrize("harness", ["react", "claude_code"])
async def test_linux_transport_and_mock_evaluation(docker_pods, monkeypatch, tmp_path, harness):
    """Run real SSH tools, persisted root edits, a clean restart, and the real Inspect scorer on a CPU fixture."""
    [environments] = [await RunPodSandbox.sample_init("transport-test", None, {})]
    env = environments["default"]
    try:
        connection = await env.connection()
        key_folder = Path(env.ssh_folder.name)
        assert (key_folder / "id_ed25519").stat().st_mode & 0o777 == 0o600
        connected = await asyncio.to_thread(
            subprocess.run,
            [*shlex.split(connection.command), "printf diagnostic-connection"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert connected.returncode == 0, connected.stderr
        assert connected.stdout == "diagnostic-connection"
        await env.write_file("nested/file.bin", b"\x00hello")
        assert await env.read_file("nested/file.bin", text=False) == b"\x00hello"
        result = await env.exec(
            [
                "python3",
                "-c",
                "import os,sys; print(os.getcwd(),os.environ['TEST_ENV_MARKER']); print(sys.stdin.read())",
            ],
            input="stdin data",
            timeout=10,
        )
        assert (
            result.success
            and "preserved" in result.stdout
            and "stdin data" in result.stdout
        )
        with pytest.raises(FileNotFoundError):
            await env.read_file("does-not-exist")
        with pytest.raises(IsADirectoryError):
            await env.read_file("nested")
        with monkeypatch.context() as limits:
            limits.setattr(SandboxEnvironmentLimits, "MAX_READ_FILE_SIZE", 3)
            with pytest.raises(OutputLimitExceededError):
                await env.read_file("nested/file.bin")
        with pytest.raises(TimeoutError):
            await env.exec(["sleep", "5"], timeout=1)
        assert not (await env.exec(["false"], timeout=5)).success

        # Reproduce a disk that fits the submission but cannot hold a second copy.
        capacity = """set -eu
mkdir -p /inferencebench-capacity/source /inferencebench-capacity/staging
dd if=/dev/urandom of=/inferencebench-capacity/source/data bs=1M count=20 status=none
sha256sum /inferencebench-capacity/source/data > /tmp/capacity.sha256
tar --format=pax --incremental --numeric-owner --sparse -C /inferencebench-capacity/source -cf /tmp/capacity.tar .
if tar -xf /tmp/capacity.tar -C /inferencebench-capacity/staging 2>/tmp/capacity-error; then exit 1; fi
python3 -c 'import os; assert os.statvfs("/inferencebench-capacity").f_bavail == 0'
rm -rf /inferencebench-capacity/staging
tar --incremental --numeric-owner -xpf /tmp/capacity.tar -C /inferencebench-capacity/source
sha256sum -c /tmp/capacity.sha256
"""
        result = await env.exec(["bash", "-c", capacity], timeout=30)
        assert result.success, result.stdout + result.stderr

        # A background compound-list shell retains pipes even when its child redirects output.
        command = "cd /tmp && nohup sleep 30 >/tmp/background.log 2>&1 &\nprintf foreground; printf diagnostic >&2; exit 7"
        result = await asyncio.wait_for(env.exec(["bash", "-c", command]), timeout=5)
        assert (result.returncode, result.stdout, result.stderr) == (7, "foreground", "diagnostic")
        with monkeypatch.context() as limits:
            limits.setattr(SandboxEnvironmentLimits, "MAX_EXEC_OUTPUT_SIZE", 1024)
            result = await env.exec([
                "python3", "-c",
                "import sys; sys.stdout.write('x'*3000000+'stdout-tail'); sys.stderr.write('y'*3000000+'stderr-tail')",
            ], timeout=10)
            assert len(result.stdout) == len(result.stderr) == 1024
            assert result.stdout.endswith("stdout-tail")
            assert result.stderr.endswith("stderr-tail")
            noise = "import os; exec(\"while True: os.write(1, b'x'*65536)\")"
            command = f"python3 -c {shlex.quote(noise)} &\nsleep 0.05; printf diagnostic >&2; exit 3"
            # Exercise interleaving between a noisy descendant and the shell exit record.
            for _ in range(20):
                result = await asyncio.wait_for(env.exec(["bash", "-c", command]), timeout=5)
                assert result.returncode == 3
                assert len(result.stdout) <= 1024
                assert result.stderr == "diagnostic"
    finally:
        await RunPodSandbox.sample_cleanup("transport-test", None, environments, False)
    assert not docker_pods.pods
    assert not key_folder.exists()

    task = inference_bench(
        gpu_provider="runpod",
        scenarios="A",
        seed_pairs=[[21, 1337]],
        request_limit=1,
        quality_samples=16, quality_cache=None,
    )
    task.solver = (
        react_agent(nudge_prompt=False, token_budget_reminder=False)
        if harness == "react" else as_solver(cli_agent(
            "claude_code", {"version": "2.1.114", "permission_mode": "bypassPermissions", "retry_refusals": 0},
            nudge_prompt=False, token_budget_reminder=False,
        ))
    )
    command = """printf preserved > /usr/local/lib/agent-change
ln /usr/local/lib/agent-change /usr/local/lib/agent-hardlink
ln -s agent-change /usr/local/lib/agent-symlink
truncate -s 1G /usr/local/lib/agent-sparse
python3 -c 'import os; os.utime("/usr/local/lib/agent-change", ns=(1700000000123456789, 1700000000123456789))'
python3 /home/agent/task/evaluate.py
python3 -c 'import json; from pathlib import Path; rows=[json.loads(line) for line in Path("/home/agent/task/requests.jsonl").read_text().splitlines()]; assert len(rows)==1; assert rows[0]["ignore_eos"]'
python3 -c 'import os; from pathlib import Path; p=Path("/usr/local/lib/same-stat"); s=p.stat(); p.write_text("modified"); os.utime(p, ns=(s.st_atime_ns, s.st_mtime_ns))'
rm /usr/local/lib/delete-me
touch /run/agent-ephemeral
printf changed > /opt/inferencebench/src/eval/inference/bin/launch_supervised_server.sh
python3 -u -c 'import time; exec("while True: print(123); time.sleep(0.001)")' >/tmp/live-writer.log 2>&1 </dev/null &
cd /home/agent/task && python3 -m http.server 8765 --bind 127.0.0.1 >/tmp/old-server.log 2>&1 </dev/null &
python3 -c 'import time,urllib.request; time.sleep(1); print(urllib.request.urlopen("http://127.0.0.1:8765").status)'
"""
    subject = get_model(
        "mockllm/subject",
        custom_outputs=[
            ModelOutput.for_tool_call("mockllm/subject", "bash" if harness == "react" else "Bash", {"command": command}),
            ModelOutput.from_content("mockllm/subject", "Ready"),
        ],
    )
    judge = get_model(
        "mockllm/judge",
        custom_outputs=[
            ModelOutput.from_content(
                "mockllm/judge", "no contamination detected\nonly allowed use detected"
            )
        ],
    )
    [log] = await eval_async(
        task,
        model=subject,
        model_roles={"integrity": judge},
        log_dir="logs",
    )
    assert log.status == "success", log.error
    assert not any(
        event.event == "logger"
        and (event.message.name or "").startswith("asyncssh")
        and event.message.level in {"debug", "info"}
        for event in log.samples[0].events
    )
    score = log.samples[0].scores["inference_speedup"]
    assert score.value == {"speedup": 2.0}
    assert (Path(log.samples[0].store["artifacts"]) / "quality-baseline.tar.gz").is_file()
    assert not docker_pods.pods
    assert any(path.endswith("/restart") for method, path in docker_pods.calls)
    assert sum(path.endswith("/restart") for method, path in docker_pods.calls) == 1
