FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive UV_HTTP_TIMEOUT=600 HF_HOME=/opt/hf_cache HOME=/home/agent

COPY assets/scripts/setup_environment.sh /opt/setup_environment.sh

RUN bash /opt/setup_environment.sh

ENV PATH=/home/agent/.local/bin:$PATH \
    PYTHONPATH=/opt/inferencebench/src/eval \
    INFERENCE_BENCH_BASE_MODEL=mistralai/Mistral-7B-Instruct-v0.3 \
    INFERENCE_BENCH_MAX_MODEL_LEN=32768 \
    INFERENCE_BENCH_ALLOW_HF_DOWNLOAD=1

WORKDIR /home/agent/task

COPY assets/scripts/runpod_start.sh /opt/runpod_start.sh

CMD ["sleep", "infinity"]
