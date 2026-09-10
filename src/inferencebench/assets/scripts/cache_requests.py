"""Build the packaged request prefixes once on a CPU machine using the original sampler."""

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from transformers import AutoTokenizer

from inferencebench.dataset import SCENARIOS
from inferencebench.run_config import load_config


def main():
    """Freeze ten requests per scenario and seed, with the source and tokenizer provenance."""
    defaults = load_config()["task"]["args"]
    original = load_config("run_configs/original.yaml")["task"]["args"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=defaults["request_limit"])
    args = parser.parse_args()
    sys.path.insert(0, str(args.upstream / "src/eval"))
    from inference import runner

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # The upstream counter silently falls back if chat-template dependencies are missing.
    tokenizer.apply_chat_template(
        [{"role": "user", "content": "Verify chat-template dependencies."}],
        tokenize=True,
        add_generation_prompt=True,
    )
    counts = {}
    count_tokens = runner._count_chat_tokens

    def cached_count(messages, tokenizer):
        """Avoid retokenizing the same full documents while generating multiple frozen workloads."""
        digest = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).digest()
        if digest not in counts:
            counts[digest] = count_tokens(messages, tokenizer)
        return counts[digest]

    runner._count_chat_tokens = cached_count
    with tempfile.TemporaryDirectory(prefix="inferencebench-cache-") as temporary:
        source = Path(temporary) / "samples.jsonl"
        with gzip.open(args.samples, "rb") as reader, source.open("wb") as writer:
            shutil.copyfileobj(reader, writer)

        def samples_file(*args, **kwargs):
            """Use the complete 503-row pool, whose reservoir selection is identical across seeds."""
            return source

        runner._find_longbench_samples_file = samples_file
        requests = {}
        for scenario, record in SCENARIOS.items():
            requests[scenario] = {}
            for seed in sorted({seed for pair in original["seed_pairs"] for seed in pair}):
                config = {**record["config"], "dataset_seed": seed}
                rows, _ = runner._prepare_requests(
                    config, args.count, tokenizer, defaults["max_model_len"]
                )
                requests[scenario][str(seed)] = rows
                print(f"Cached {scenario}/{seed}: {len(rows)} requests", flush=True)
        with source.open("rb") as reader:
            pool_hash = hashlib.file_digest(reader, "sha256").hexdigest()

    payload = json.dumps(requests, sort_keys=True, ensure_ascii=False).encode()
    cache = {
        "format_version": 1,
        "base_model": defaults["base_model"],
        "max_model_len": defaults["max_model_len"],
        "upstream_revision": subprocess.check_output(
            ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True
        ).strip(),
        "dataset": "https://huggingface.co/datasets/zai-org/LongBench-v2",
        "source_pool_sha256": pool_hash,
        "tokenizer_files_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(args.tokenizer.iterdir()) if path.is_file()
        },
        "scenario_config_sha256": hashlib.sha256(
            json.dumps(SCENARIOS, sort_keys=True).encode()
        ).hexdigest(),
        "requests_sha256": hashlib.sha256(payload).hexdigest(),
        "requests": requests,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(gzip.compress(json.dumps(cache, ensure_ascii=False).encode(), mtime=0))
    print(f"Wrote {args.output}: {args.output.stat().st_size} bytes", flush=True)


if __name__ == "__main__":
    main()
