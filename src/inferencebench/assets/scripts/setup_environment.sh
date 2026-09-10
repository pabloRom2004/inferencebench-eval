#!/bin/bash
set -euo pipefail

# Build tools and the original CUDA/PyTorch stack.
apt-get update && apt-get install -y \
    python3 python3-dev python3-pip python3-venv git curl wget \
    iproute2 net-tools lsof procps ripgrep ninja-build pkg-config cmake \
    build-essential openssh-server rsync
rm -rf /var/lib/apt/lists/*
pip install uv
uv pip install --system torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --system \
    aiohttp requests accelerate datasets sentencepiece protobuf numpy tokenizers 'transformers<5'
curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
apt-get install -y nodejs

# Pin the evaluator source and cache the target model.
git clone https://github.com/aisa-group/InferenceBench.git /opt/inferencebench
(cd /opt/inferencebench && git checkout 24cdf88f6a4e14ed85d665aa132cecccb3ee95ef)
mkdir -p /home/agent/task /opt/inference_eval
cp -r /opt/inferencebench/src/eval/inference/bin /opt/inference_eval/bin
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('mistralai/Mistral-7B-Instruct-v0.3', ignore_patterns=['*.pt', '*.bin', 'original/*'])"

# Isolate evaluator dependencies from agent-installed serving engines.
uv venv /opt/evaluator
uv pip install --python /opt/evaluator/bin/python \
    aiohttp requests datasets sentencepiece protobuf numpy jinja2 'transformers<5'
