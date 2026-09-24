"""Kill a real container mid-run and resume every harness family from its checkpoint on a fresh one (no GPU or paid API)."""

import asyncio
import importlib
import json
import os
import re
import subprocess
import uuid
from types import SimpleNamespace

import pytest
from inspect_ai import eval_set
from inspect_ai.agent import as_solver
from inspect_ai.log import read_eval_log
from inspect_ai.model import GenerateConfig, ModelOutput, ModelUsage, get_model
from inspect_ai.util import sample_limits

from inferencebench import cli_agent, inference_bench, original_agent, react_agent
from inferencebench.environment import RunPodSandbox, record_checkpoint_time
from inferencebench.prompts import ASSETS, CHECKPOINT_RESUME
from tests.inferencebench.test_task import fake_judge_cli as fake_judge_cli
from tests.inferencebench.test_task import judge_model

SHELL_TOOLS = {"bash", "exec", "exec_command", "shell", "run_shell_command"}
# The versions verified for this port and the recovery tests they share.
CLI_ARGS = {
    "claude_code": {
        "version": "2.1.267",
        "permission_mode": "bypassPermissions",
        "retry_refusals": 0,
    },
    "codex_cli": {"version": "0.154.0", "web_search": "disabled"},
    "gemini_cli": {"version": "0.60.0"},
    "kimi_code": {"version": "2.0.0"},
    "opencode": {"version": "1.18.31"},
}
EVALUATOR = '''#!/usr/bin/python3
"""Return synthetic measurements, and check that final scoring sees the restored workspace."""
import json, sys, tarfile
from pathlib import Path
folder = Path('/tmp/inferencebench')
operation = sys.argv[2]
metrics = {'profiles': {'burst': {'success_count': 1, 'ttft': {'p50': 2 if operation == 'prepare' else 1}}}, 'quality_check': {'pass': True}}
if operation == 'install':
    with tarfile.open(folder / 'upstream.tar') as archive:
        timer = archive.extractfile('upstream/src/utils/create_timer.sh').read()
    Path('/opt/inferencebench/src/utils').mkdir(parents=True, exist_ok=True)
    Path('/opt/inferencebench/src/utils/create_timer.sh').write_bytes(timer)
elif operation == 'prepare':
    trusted = folder / 'trusted'
    (trusted / 'speed').mkdir(parents=True, exist_ok=True)
    (trusted / 'quality').mkdir(exist_ok=True)
    (trusted / 'speed/requests.jsonl').write_text('{}\\n')
    (trusted / 'speed/baseline_metrics.json').write_text(json.dumps({'baseline': metrics}))
    (trusted / 'quality/samples.jsonl').write_text('{}\\n')
    (trusted / 'provenance.json').write_text(json.dumps({'downloaded_model_revision': 'cpu-fixture'}))
    (trusted / 'environment.json').write_text(json.dumps({'INFERENCE_BENCH_BASE_MODEL': 'fixture'}))
    Path('/home/agent/task/start_server.sh').touch()
else:
    assert Path('/home/agent/task/notes.txt').read_text().startswith('restore-marker-')
    (folder / 'final.json').write_text(json.dumps(metrics))
    (folder / 'final-server.log').write_text('CPU fixture: restored workspace reached final scoring')
'''


@pytest.fixture
async def pods(monkeypatch, tmp_path):
    """Serve RunPod's API from local containers, with real SSH, startup, snapshot, and restart code."""
    if os.environ.get("INFERENCEBENCH_TEST_DOCKER") != "1":
        pytest.skip(
            "Set INFERENCEBENCH_TEST_DOCKER=1 for the local container restore test"
        )
    image = "inferencebench-restore-test:" + uuid.uuid4().hex[:12]
    context = tmp_path / "image"
    (context / "scripts").mkdir(parents=True)
    for name in ["runpod_start.sh", "runtime.py"]:
        (context / "scripts" / name).write_bytes(
            (ASSETS / "scripts" / name).read_bytes()
        )
    (context / "scripts" / "setup_environment.sh").write_text(
        "echo setup >> /workspace/setup-calls\n"
    )
    (context.parent / "Dockerfile").write_text(
        (ASSETS.parent / "Dockerfile").read_text()
    )
    monkeypatch.setattr(
        importlib.import_module("inferencebench.environment"), "ASSETS", context
    )
    (context / "evaluator.py").write_text(EVALUATOR)
    (context / "Dockerfile").write_text("""FROM ubuntu:22.04
RUN apt-get update && apt-get install -y openssh-server rsync python3 curl git ca-certificates && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /home/agent/task /opt/evaluator/bin /opt/inference_eval/bin
COPY scripts/runpod_start.sh /opt/runpod_start.sh
COPY evaluator.py /opt/evaluator/bin/python
RUN chmod +x /opt/evaluator/bin/python && printf '#!/bin/sh\\necho "CPU-only synthetic GPU fixture"\\n' > /usr/local/bin/nvidia-smi && chmod +x /usr/local/bin/nvidia-smi
ENV HF_HOME=/opt/hf_cache
ENTRYPOINT ["bash", "/opt/runpod_start.sh"]
""")
    build = await asyncio.to_thread(
        subprocess.run,
        ["docker", "build", "-t", image, str(context)],
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    live = {}
    created = []

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
        """Start a pristine container on the pod's volume, as RunPod does after creation or restart."""
        args = ["run", "-d", "-p", "127.0.0.1::22", "-v", pod["volume"] + ":/workspace"]
        for name, value in pod["env"].items():
            args += ["-e", name + "=" + value]
        pod["container"] = await docker(
            *args, "--entrypoint", "bash", image, "-c", pod["command"]
        )

    async def remove(pod_id):
        """Delete a pod's container and volume, as a RunPod pod deletion does."""
        pod = live.pop(pod_id)
        await docker("rm", "-f", pod["container"])
        await docker("volume", "rm", pod["volume"])

    async def request(self, method, path, **kwargs):
        """Map RunPod REST operations onto local containers; a deleted pod answers 404."""
        if method == "POST" and path == "/pods":
            pod_id = uuid.uuid4().hex
            body = kwargs["json"]
            pod = {
                "env": body["env"],
                "volume": "inferencebench-restore-" + pod_id,
                "command": body["dockerStartCmd"][0],
            }
            live[pod_id] = pod
            created.append(pod_id)
            await docker("volume", "create", pod["volume"])
            await start(pod)
            return {"id": pod_id}
        pod_id = path.split("/")[2]
        if method == "DELETE":
            if pod_id in live:
                await remove(pod_id)
            return None
        pod = live[pod_id]
        if method == "POST":
            await docker("rm", "-f", pod["container"])
            await start(pod)
            return None
        record = json.loads(await docker("inspect", pod["container"]))[0]
        port = record["NetworkSettings"]["Ports"]["22/tcp"][0]["HostPort"]
        return {"publicIp": "127.0.0.1", "portMappings": {"22": int(port)}}

    def kill():
        """Delete the only live pod from underneath a running sample, like an operator or host failure."""
        [(pod_id, pod)] = live.items()
        del live[pod_id]
        subprocess.run(
            ["docker", "rm", "-f", pod["container"]], check=True, capture_output=True
        )
        subprocess.run(
            ["docker", "volume", "rm", pod["volume"]], check=True, capture_output=True
        )
        return pod_id

    monkeypatch.setenv("RUNPOD_API_KEY", "test-key-not-a-credential")
    monkeypatch.setenv("RUNPOD_IMAGE", image)
    monkeypatch.setattr(RunPodSandbox, "_request", request)
    # Keep this fixture's pod records, baselines, and artifacts out of the repository.
    monkeypatch.chdir(tmp_path)
    try:
        yield SimpleNamespace(live=live, created=created, kill=kill)
    finally:
        for pod_id in list(live):
            await remove(pod_id)
        await docker("image", "rm", "-f", image)


def solver_for(harness):
    """Build the harness under test the way the maintained and original configs do."""
    if harness == "react":
        return react_agent(nudge_prompt=False, token_budget_reminder=False)
    if harness == "original":
        return original_agent(
            "claude_code", "2.1.267", continue_until_deadline=False, env={}
        )
    return as_solver(
        cli_agent(
            harness,
            dict(CLI_ARGS[harness]),
            nudge_prompt=False,
            token_budget_reminder=False,
        )
    )


@pytest.mark.parametrize("harness", ["react", "original", *CLI_ARGS])
def test_container_death_resumes_from_checkpoint(pods, tmp_path, harness):
    """Restore files, native session, usage, and deadline on a fresh container, then score the resumed work."""
    marker = "restore-marker-" + uuid.uuid4().hex
    seen = {"phase": "work", "saved_usage": [], "resumed_calls": 0}
    checkpoints = tmp_path / "evals"

    def output(messages, tools, tool_choice, config):
        """Write files until a checkpoint commits, die, then verify what the restored session sees."""
        if not tools:
            # Native title and summary requests run beside the task conversation.
            return ModelOutput.from_content(
                "mockllm/subject", "Checkpoint restore test"
            )
        text = "\n".join(message.text for message in messages)
        if seen["phase"] == "work" and not list(checkpoints.rglob("ckpt-*.json")):
            # Each two-second tool outlasts the one-second trigger, so a later boundary saves.
            command = f"echo {marker} | tee /home/agent/task/notes.txt; touch /opt/outside-marker; sleep 2"
        elif seen["phase"] == "work":
            seen["killed"], seen["phase"] = pods.kill(), "resume"
            command = "echo lost-after-checkpoint"
        elif seen["phase"] == "resume":
            # Hydration has restored the saved usage before this request is counted.
            seen["restored_usage"] = sample_limits().token.usage
            seen["restored_text"], seen["phase"] = text, "verify"
            seen["restored_lost_call"] = any(
                "lost-after-checkpoint" in str(call.arguments)
                for message in messages
                if message.role == "assistant"
                for call in message.tool_calls or []
            )
            command = (
                # Gemini CLI refuses command substitution.
                "printf verify:; cat /home/agent/task/notes.txt; test -e /opt/outside-marker && echo installs-survived || echo installs-gone; "
                "grep -E '^(CREATION_DATE|NUM_SECONDS)=' /home/agent/task/timer.sh"
            )
        else:
            seen["verified_text"] = "\n".join(
                message.text for message in messages if message.role == "tool"
            )
            seen["resumed_calls"] += 1
            result = ModelOutput.from_content(
                "mockllm/subject", "Restored work is complete."
            )
            result.usage = ModelUsage(
                input_tokens=90, output_tokens=10, total_tokens=100
            )
            return result
        if seen["phase"] == "verify":
            seen["resumed_calls"] += 1
        shell = next(tool for tool in tools if tool.name.lower() in SHELL_TOOLS)
        arguments = {
            ("cmd" if "cmd" in shell.parameters.properties else "command"): command
        }
        if "description" in shell.parameters.required:
            arguments["description"] = "Checkpoint restore test"
        result = ModelOutput.for_tool_call("mockllm/subject", shell.name, arguments)
        result.usage = ModelUsage(input_tokens=90, output_tokens=10, total_tokens=100)
        return result

    async def saved(state):
        """Record the usage each first-attempt checkpoint captures, then keep the task's own timestamp."""
        if seen["phase"] == "work":
            seen["saved_usage"].append(sample_limits().token.usage)
        await record_checkpoint_time(state)

    task = inference_bench(
        gpu_provider="runpod",
        scenarios="A",
        seed_pairs=[[21, 1337]],
        quality_samples=16,
        agent_seconds=3600,
        checkpoint_seconds=1,
        # Kimi Code and OpenCode size their native context from the served model, which mockllm does not describe.
        context_length=128000,
    )
    task.solver = solver_for(harness)
    task.on_checkpoint = saved
    success, logs = eval_set(
        task,
        # OpenCode also needs an output limit to size its compaction.
        model=get_model(
            "mockllm/subject",
            config=GenerateConfig(max_tokens=8192),
            custom_outputs=output,
            memoize=False,
        ),
        model_roles={"integrity": judge_model()},
        log_dir=str(checkpoints),
        retry_attempts=2,
        retry_wait=1,
        retry_immediate=False,
        retry_cleanup=False,
        display="none",
    )
    attempts = sorted(
        (read_eval_log(path) for path in checkpoints.glob("*.eval")),
        key=lambda log: log.eval.created,
    )
    assert success, [
        log.samples[0].error for log in attempts if log.samples and log.samples[0].error
    ]
    failed, final = attempts[0].samples[0], read_eval_log(logs[0].location).samples[0]
    assert len(attempts) == 2 and failed.error is not None and not failed.scores
    assert final.error is None
    assert final.scores["inference_speedup"].value == {"speedup": 2.0}

    # A fresh pod, with the controller's cached speed baseline reused by the second preparation.
    assert failed.metadata["agent_sandbox_id"] == seen["killed"]
    assert final.metadata["agent_sandbox_id"] not in (None, seen["killed"])
    assert final.metadata["speed_baseline"]["source"] == "cache"
    assert CHECKPOINT_RESUME.prompt in final.input

    # Conversation and native session came back without the unsaved tail; workspace files did, installs did not.
    assert marker in seen["restored_text"] and not seen["restored_lost_call"]
    verified = seen["verified_text"]
    assert f"verify:{marker}" in verified
    assert "installs-gone" in verified and "installs-survived" not in verified

    # Usage continues from the last first-attempt checkpoint, plus the resumed turns.
    assert seen["restored_usage"] == seen["saved_usage"][-1] > 0
    assert (
        final.token_limit_usage == seen["restored_usage"] + 100 * seen["resumed_calls"]
    )

    # The deadline moves by exactly the downtime, and the restored timer reports the extended deadline.
    [report] = [
        event.data["report"]["data"]
        for event in final.events
        if event.event == "info"
        and event.source == "checkpoint"
        and event.data.get("event") == "resume"
    ]
    assert report["attempt"] == "resume" and report["downtime_seconds"] > 0
    assert report["deadline"] == pytest.approx(
        report["previous_deadline"] + report["downtime_seconds"]
    )
    assert final.store["deadline"] == pytest.approx(report["deadline"])
    # Some CLIs return tool output inside JSON with escaped newlines.
    timer = dict(re.findall(r"(CREATION_DATE|NUM_SECONDS)=(\d+)", verified))
    assert int(timer["NUM_SECONDS"]) < 3600
    assert int(timer["CREATION_DATE"]) + int(timer["NUM_SECONDS"]) == pytest.approx(
        report["deadline"], abs=3
    )
    assert not pods.live
