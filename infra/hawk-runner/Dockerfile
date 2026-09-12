FROM ghcr.io/astral-sh/uv:0.12.3@sha256:2d890623d310b57771ce840f0da5eed5fc6d657da05ffaa45d82797b53fa3abc AS uv
FROM python:3.13-slim-bookworm@sha256:ed86c82274b3c69b52fb5820f358f0bd7df0b603332063cb5c6e32bd220c3e6e

RUN apt-get update && apt-get install -y --no-install-recommends git libjemalloc2 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 65532 --create-home nonroot
COPY --from=uv /uv /uvx /usr/local/bin/
RUN uv pip install --system 'hawk[runner]==2.5.0' 'inspect-ai==0.3.263'
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 UV_LINK_MODE=copy
ENV LD_PRELOAD=libjemalloc.so.2
WORKDIR /home/nonroot
USER 65532
STOPSIGNAL SIGINT

# The stock Helm template overrides runner.environment with PATCH_SANDBOX=true.
# Keep the RunPod backend; Hawk still rejects nonstandard isolation without patching.
ENTRYPOINT ["env", "HAWK_RUNNER_PATCH_SANDBOX=false", "python", "-m", "hawk.runner.entrypoint"]
