"""Run a deliberately small real-H100 transport and evaluator smoke test, never a headline benchmark result."""

import argparse
import json
from pathlib import Path

from inspect_ai import eval as inspect_eval
from inspect_ai.model import ModelOutput, get_model

from inferencebench import inference_bench, original_agent, react_agent
from inferencebench.run_config import load_config

SCRIPT = """nvidia-smi --query-gpu=name,memory.total --format=csv
id
cat > /home/agent/task/start_server.sh <<'SH'
#!/bin/bash
set -euo pipefail
exec python3 /opt/inferencebench/src/eval/inference/servers/transformers_openai_server.py --model mistralai/Mistral-7B-Instruct-v0.3 --port 8000 --dtype float16 --max-model-len 32768
SH
chmod +x /home/agent/task/start_server.sh
python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Matrix product:', (torch.ones(32,32,device='cuda') @ torch.ones(32,32,device='cuda'))[0,0].item())"
"""


def main():
    """Exercise genuine ReAct tools, GPU inference, a fresh process environment, and the final scorer."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", action="store_true")
    parser.add_argument(
        "--gpu-provider",
        choices=["modal", "runpod"],
        default=load_config()["task"]["args"]["gpu_provider"],
    )
    parser.add_argument("--gpu-config", default=None)
    args = parser.parse_args()
    task = inference_bench(
        gpu_provider=args.gpu_provider,
        gpu_config=args.gpu_config,
        scenarios="A",
        seed_pairs=[[21, 1337]],
        request_limit=1,
        quality_samples=16,
        quality_cache=None,
        agent_seconds=180,
    )
    task.solver = (
        original_agent(continue_until_deadline=False)
        if args.original
        else react_agent(nudge_prompt=False, token_budget_reminder=False)
    )
    subject = get_model(
        "mockllm/subject",
        custom_outputs=[
            ModelOutput.for_tool_call(
                "mockllm/subject",
                "Bash" if args.original else "bash",
                {"command": SCRIPT},
            ),
            ModelOutput.from_content(
                "mockllm/subject", "The reproducible Transformers launcher is ready."
            ),
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
    [log] = inspect_eval(
        task,
        model=subject,
        model_roles={"integrity": judge},
        max_samples=1,
        log_dir="logs",
        display="plain",
    )
    report = Path("run-artifacts/port-validation") / (
        f"{args.gpu_provider}-smoke" + ("-original" if args.original else "") + ".json"
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(
            {"status": log.status, "log": log.location, "error": str(log.error)},
            indent=2,
        )
    )
    if log.status != "success" or not log.samples[0].scores:
        raise RuntimeError(f"{args.gpu_provider} smoke failed: {log.error}")
    sample = log.samples[0]
    assert sample.metadata["provenance"]["downloaded_model_revision"]
    assert len(sample.metadata["provenance"]["input_sha256"]) == 3
    score = sample.scores["inference_speedup"]
    assert score.metadata["final"]["quality_check"]["pass"]
    assert all(
        profile["success_count"] > 0
        for profile in score.metadata["final"]["profiles"].values()
    )
    assert sample.metadata["gpu_provider"] == args.gpu_provider
    if args.gpu_provider == "modal":
        assert (
            sample.metadata["modal_sandbox_id"] != sample.metadata["scoring_sandbox_id"]
        )
    assert any(
        message.role == "tool" and "Matrix product: 32.0" in message.text
        for message in sample.messages
    )
    print(log.samples[0].scores)


if __name__ == "__main__":
    main()
