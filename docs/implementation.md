# InferenceBench implementation guide

[InferenceBench](https://arxiv.org/abs/2607.20468) evaluates agents that deploy and optimize an inference server for Mistral-7B-Instruct-v0.3 on one H100 80GB. The original benchmark gives each agent two hours, root access, the Internet, cached model weights, and an OpenAI-compatible serving contract.

This Inspect port vendors the [original repository](https://github.com/aisa-group/InferenceBench/tree/24cdf88f6a4e14ed85d665aa132cecccb3ee95ef) byte-for-byte under `src/inferencebench/upstream/`, installs it into every sandbox with the patches in `src/inferencebench/patches/`, and drives its own entrypoints: the sampler caches, baseline and reference precompute scripts, task `evaluate.py`, launch scaffold, timer, task prompt, quality parser, and integrity rubric. The default solver is Inspect ReAct with 100 million input-plus-output tokens per attempt and no optimization deadline. A separate original configuration runs Claude Code 2.1.114 through Inspect SWE and resumes early exits until the time budget expires.

## Usage

### Installation

Install from this checkout with `uv sync`. Python 3.12 or later is required on the host; the remote environment uses Ubuntu 22.04, CUDA 12.8, Python 3.10, and PyTorch 2.8.0. Modal builds the image remotely. RunPod installs the same shared setup script on its public CUDA base image at first boot, or accepts a prebuilt registry image. Neither default path requires a local Docker daemon.

Authenticate with `uv run modal token new` and select the intended `MODAL_PROFILE`. Provider credentials stay in the local Inspect process. The Modal sandbox receives neither the Modal token nor the model-provider key.

### Running evaluations

Run one scenario and seed pair first:

```bash
uv run inspect eval --run-config src/inferencebench/run_configs/default.yaml \
  --model openrouter/anthropic/claude-sonnet-4.6 \
  -T scenarios=A -T 'seed_pairs=[[21,1337]]'
```

The default config selects scenario A with one development/evaluation seed pair. Set `task.args.scenarios: null` to run all four scenarios, one sample each. The original config retains three pairs per scenario, giving twelve samples. Both schedule one sample at a time. Each sample incurs GPU charges for speed-baseline preparation, optimization, final measurement, and integrity judging. Each sample also measures the 500-question MMLU-Pro reference before optimization: minutes with the default vLLM reference server, about 57 minutes with the original Transformers server. The default permits 100 million tokens per sample; pass `--token-limit` to select a smaller run budget. Baseline preparation happens before optimization begins. The original configuration starts its two-hour timer after preparation.

### Configurations

The original CLI configuration uses the same task and scorer:

```bash
uv run inspect eval --run-config src/inferencebench/run_configs/original.yaml \
  --model openrouter/anthropic/claude-sonnet-4.6 \
  -T scenarios=A -T 'seed_pairs=[[21,1337]]'
```

Both configs support [Inspect SWE](https://meridianlabs-ai.github.io/inspect_swe/) harnesses. `claude_code.yaml` and `codex_cli.yaml` are the maintained workload with `inferencebench/cli_agent` and the verified native settings (Claude Code 2.1.267, Codex 0.154.0, a 7200-second remote poll timeout, Codex's native search disabled in favour of the bridged tool). For another native CLI, replace the solver:

```bash
uv run inspect eval --run-config src/inferencebench/run_configs/default.yaml \
  --model openrouter/anthropic/claude-sonnet-4.6 \
  --solver inspect_swe/claude_code \
  -S cwd=/home/agent/task -S user=root -S version=auto
```

Replace `claude_code` with `codex_cli`, `opencode`, or `gemini_cli`. `--model` selects the model behind the CLI; `-S` passes the selected Inspect SWE agent's options, such as `version`, `env`, or CLI-specific settings. `version=auto` uses an installed CLI or downloads Inspect SWE's default version; use an explicit version for reproducibility. The CLI runs in the same GPU sandbox, with shared preparation and final grading. Modal remains the default; `-T gpu_provider=runpod` still works.

To switch from `inferencebench/react_agent` to Claude Code in YAML, use the full name `inspect_swe/claude_code` and replace the **entire** `solver` block, including the ReAct-specific arguments:

```yaml
solver:
  solver: inspect_swe/claude_code
  args:
    cwd: /home/agent/task
    user: root
    version: auto
```

Native CLIs use their own tools and stopping behavior, so they **can finish before exhausting the token budget**. Inspect's token cap still applies, but ReAct's nudges and reminders do not. Keep `attempts: 1` (Inspect SWE's default): additional attempts invoke this benchmark's final scorer between attempts, restarting the submission and exposing final feedback. For the original two-hour continuation behavior, keep the original wrapper and change its harness instead:

```bash
uv run inspect eval --run-config src/inferencebench/run_configs/original.yaml \
  --model openrouter/anthropic/claude-sonnet-4.6 \
  --solver inferencebench/original_agent -S harness=codex_cli -S version=auto \
  -T scenarios=A -T 'seed_pairs=[[21,1337]]'
```

For normal Python use:

```python
from inspect_ai import eval
from inferencebench import inference_bench

eval(inference_bench(scenarios="A", seed_pairs=[[21, 1337]]),
     model="openrouter/anthropic/claude-sonnet-4.6", max_samples=1)
```

You can also run `uv run inspect eval inferencebench/inference_bench --model ... --max-samples 1`. The bare task uses the default YAML's task, solver, generation, epoch, and supported task-limit settings. `limit` and `max_samples` are evaluation-layer settings: use the run config or pass them to Inspect explicitly. Native `--solver`, `-S`, `--model-role`, `--epochs`, and generation overrides remain available. Setup and final grading also run when the solver is replaced.

Inspect logs go into the flat `logs/` directory; raw measurements, evaluator output, and sandbox IDs go into `run-artifacts/inferencebench/`. View transcripts with `uv run inspect view`.

## Environment and agent options

Both solvers can install vLLM, SGLang, TGI, or another engine; read and modify their source; compile CUDA/Triton kernels; change batching, KV-cache allocation, attention backends, quantization, and launch flags; and test repeatedly with `evaluate.py`. vLLM is not imposed by the task. The initial `start_server.sh` is the original engine-neutral stub.

ReAct exposes Inspect's bash and Python tools plus a `web_search` tool, and compacts at 75% of context. Search uses [DDGS](https://github.com/deedy5/ddgs) with DuckDuckGo, requires no search API key, and works with OpenRouter models including GLM. It returns titles, URLs, and snippets; agents can fetch full pages with their shell tools. Search runs on the host and its calls and results appear in the Inspect transcript. `solver.args.web_search_args` configures the backend, result count, and request timeout; remove `web_search` from `solver.args.tools` to disable it. Public search services may rate-limit requests; these return tool errors that the agent can recover from.

By default, `submit: false` removes the submit tool, `nudge_prompt: true` resumes every early final answer without a nudge-count limit, and `token_budget_reminder: true` shows the effective token limit and remaining budget before the first turn and after subsequent turns. The budget counts cumulative input and output across API calls, including cached input. There is no optimization time limit, turn limit, or extra per-response output cap. Search uses no additional language-model calls.

Token exhaustion is the default agent's only normal stopping condition. Set `submit: true` to permit explicit submission, or `nudge_prompt: false` to allow an ordinary final answer to finish. Infrastructure and provider errors remain errors; [Modal's 24-hour sandbox lifetime](https://modal.com/docs/guide/sandboxes) still applies. Native Inspect token limits are checked at generation boundaries, so the final call may cross the nominal limit. The original configuration retains Claude Code's own tools and context management. `original_agent` accepts compatible Inspect SWE factory names, including `codex_cli`, `opencode`, and `gemini_cli`; pass an appropriate CLI version explicitly when selecting another harness.

`cli_agent` shares continuation, token reminders, and host-side search with ReAct while retaining each native CLI session and compaction. Claude Code’s built-in WebSearch is replaced by that bridged search. All model requests, including auxiliary Claude Code roles, use Inspect’s configured model bridge.

The CLI agents also have outbound Internet access through their shell tools. Their built-in search tools depend on the selected CLI and [Inspect's model bridge](https://inspect.aisi.org.uk/agent-bridge.html); shell connectivity alone does not guarantee that a native search tool works with every bridged model. The added ReAct search tool does not alter `original.yaml`.

The agent must preserve a standalone `start_server.sh`. Final scoring preserves the installed filesystem, replaces or restarts the GPU container without its processes, restores trusted evaluator code and held-out inputs from the host, and launches the submission under the upstream supervisor.

### Switching GPU providers

Modal remains the default in both run configurations. To use RunPod, select `gpu_provider: runpod` or pass `-T gpu_provider=runpod`. Both providers run the same agent, evaluator, and scoring policy.

RunPod keeps SSH handshake and command-wrapper messages out of the transcript; SSH warnings and failures remain visible. Inspect still records the model's actual tool calls, results, and sandbox operations. The SSH wrapper is transport plumbing, not an additional model turn.

Set `RUNPOD_API_KEY` in the host environment, then launch:

```bash
uv run inspect eval --run-config src/inferencebench/run_configs/default.yaml \
  --model openrouter/z-ai/glm-5.3-flash \
  -T gpu_provider=runpod -T scenarios=A -T 'seed_pairs=[[21,1337]]' \
  --token-limit 500000
```

RunPod settings live in [runpod.yaml](../src/inferencebench/runpod.yaml): one H100 SXM 80GB, at least 16 vCPUs and 180 GB RAM, a 100 GB container disk, and a 100 GB attached volume. The volume holds an archive of the installed filesystem for final scoring. Copy this file and pass `-T gpu_config=/absolute/path/runpod.yaml` to change resource or startup settings. The adapter owns the pod name, SSH environment, entrypoint, and startup command.

With `bootstrap: true`, first boot executes [setup_environment.sh](../src/inferencebench/assets/scripts/setup_environment.sh), the same installation script used by the Dockerfile. This includes downloading the model and installing the pinned vLLM reference environment, and incurs GPU charges during setup. Restart restores the saved filesystem without reinstalling the environment. For repeated runs, build and push this package's Dockerfile to a registry you control, set `RUNPOD_IMAGE` or `pod.imageName` to its image reference, and set `bootstrap: false` in your provider config. `pod.containerRegistryAuthId` supports private-registry credentials already configured in RunPod.

[RunPod treats container storage as ephemeral](https://docs.runpod.io/pods/storage/types). This backend pauses background processes and writes one full archive to the attached volume, avoiding slow per-file network writes. After restart, it restores files directly into the root filesystem and verifies a new boot marker. The archive uses GNU tar's directory inventories to restore deletions, including when a restart retains the existing disk; it needs no second staging copy. The complete filesystem must fit on both configured disks. Virtual filesystems and provider runtime state under `/run` are excluded. When `/proc/self/mountinfo` confirms that `/etc/nvidia/nvidia-application-profiles-rc.d` is a separate mount, its directory inventory and metadata are excluded too. RunPod creates this protected NVIDIA runtime directory independently of the installed filesystem; trying to delete its files or restore its ownership can otherwise prevent startup. An ordinary unmounted directory keeps the normal snapshot behavior. All extraction errors remain fatal. Modal uses its filesystem snapshot API. Both paths restore the trusted evaluator and held-out files from the host afterward. H100 availability, host CPU allocation, and storage performance can differ between providers; compare each result against its own measured baseline and record the provider.

The host authenticates to the RunPod API and uses per-pod SSH keys with a pinned host key. Native SSH keepalives preserve connections during silent preparation commands. Failed connection attempts retry up to `api_retry_attempts` before sending commands; commands already sent are never replayed. Exhausted connection attempts report sandbox unavailability, preserving the normal error and cleanup path. Transient restart response failures wait for a new boot marker through the configured startup allowance before retrying, up to `api_retry_attempts`; they do not rerun the agent or filesystem snapshot. Pod creation retries only definite server rejections (HTTP 429 and 5xx, which is how RunPod reports capacity shortfalls), up to `create_retry_attempts` spaced `create_retry_interval_seconds` apart, after checking that no pod already carries the sample's unique name; a lost create response is never replayed. The adapter sends neither your RunPod account key nor your model-provider key to the pod. RunPod itself injects a separate [pod-scoped API key](https://docs.runpod.io/pods/templates/environment-variables). Completion and cancellation delete the owned pod and its attached volume. Unlike Modal's 24-hour platform cap, this RunPod adapter relies on host-side cleanup: a killed or disconnected host can leave a pod running. Pod IDs and cleanup status are recorded in `run-artifacts/runpod/`. To release a recorded pod explicitly:

```bash
uv run inspect sandbox cleanup inferencebench_runpod POD_ID
```

Once SSH is ready, the pod record also contains an `ssh_command` for inspecting the live environment from another terminal. Its private key is stored in a private temporary directory outside the repository and removed during cleanup; the command pins the pod's host key.

RunPod support is locally tested with mocked REST calls and real SSH/Linux restarts on both fresh and retained disks, including same-size edits, hardlinks, sparse files, and constrained disk space. Live GLM 5.3 Flash H100 smoke tests also exercised the cached evaluator, a 59 GB filesystem restore, quality grading, and cleanup; an earlier smoke separately exercised integrity judging. These used one or ten Scenario A requests and 16 quality items, so they do not establish full-benchmark performance. Lambda and Vast.ai are not implemented.

## Parameters

Adjustable task and solver defaults are in `src/inferencebench/run_configs/default.yaml`. `original.yaml` selects the original prompt, a two-hour deadline, the API-Claude harness variant, and no token cap. Claude Code 2.1.114 is the port’s tested version pin; upstream installed an unpinned CLI. The default changes the budget wording and uses ReAct with token reminders and continuation nudges.

The configs show common Inspect controls. Additional native options can be added under `generate_config` or `eval_config`, or passed through the CLI.

Claude Code receives each pending task or continuation message through a file
in `/tmp`, with a short instruction to read it. The instruction text is unchanged.
This avoids putting server names in Claude Code's process arguments, where a
normal `pkill -f` server restart can accidentally kill the agent. Both maintained
and original Claude Code adapters use this transport. Inspect SWE 0.2.70 and
upstream main still passed the full prompt in process arguments when checked
on 2026-09-13. Hawk task cleanup captures unfinished submissions and transcripts
before removing a failed agent's sandbox; it preserves the failure outcome.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `config_defaults` | `default` | Packaged fallback for native settings and judge roles; `original.yaml` selects `original` |
| `gpu_provider` | `modal` | GPU backend: `modal` or `runpod` |
| `gpu_config` | `null` | Optional provider YAML path; otherwise use the packaged provider config |
| `scenarios` | `A` | Scenario filter: a string, list, or `null` for all A–D |
| `seed_pairs` | `[[21,1337]]` | Development and held-out evaluation seeds; `original.yaml` retains three pairs |
| `base_model` | `mistralai/Mistral-7B-Instruct-v0.3` | Fixed model checkpoint for the benchmark |
| `max_model_len` | 32768 | Original evaluator and baseline context limit |
| `baseline_dtype` | `float16` | Precision of the Transformers speed baseline and the reference server; `bfloat16` for bf16-native models |
| `context_length` | `null` | Optimizing model's context window; used by Inspect compaction and model bridges |
| `agent_seconds` | `null` | Optional optimization wall-clock limit; `original.yaml` uses 7200 seconds |
| `eval_config.token_limit` | 100000000 | Total input-plus-output tokens per attempt; override with `--token-limit` |
| `request_limit` | 10 | Requests per load profile; `null` requests the original scenario count |
| `quality_samples` | 500 | MMLU-Pro quality-gate questions |
| `quality_seed` | 248 | Fixed quality sample seed |
| `quality_reference_backend` | `vllm` | Server measuring the MMLU-Pro reference: pinned vLLM 0.19.0 in float16, or the original `transformers` server (`original.yaml`) |
| `quality_concurrency` | 4 | Concurrent quality-gate requests, and the vLLM reference's concurrency; upstream measures the Transformers reference at 1 |
| `quality_baseline_max_attempts` | 2 | Total attempts per reference question; failed questions retry individually. `1` accepts upstream's registry as its precompute wrote it |
| `quality_tau` | 0.95 | Required fraction of the Transformers baseline accuracy |
| `server_wait_seconds` | 900 | Final server readiness allowance |
| `request_timeout_seconds` | 300 | Per-request timeout |
| `system_prompt` | `token_budget` | Original task text adapted to token budgeting; `original.yaml` uses the verbatim prompt |
| `strict_prompt` | `true` | Insert two bullets after the base-model constraint stating the leaderboard's strict rules: no third-party pre-quantized checkpoints, no modifying the evaluation harness. The authors' strict prompt is unreleased; `false` restores the paper's Table 2 prompt |
| `scorer` | `inference_speedup` | Replaceable scorer factory and judge-role options |

The integrity judge is upstream's judge: after measurement, upstream's `get_judge_prompt.py` renders the rubric plus the restarted `start_server.sh` and the last 200 lines of `final-server.log`, and Claude Code (`judge_cli_version`, installed by Inspect SWE in the scoring sandbox) runs that prompt from `/home/agent/task` with permissions bypassed, exactly like upstream's `claude --print --dangerously-skip-permissions` call. The scorer first removes any verdict files the agent may have left, then reads `contamination_judgement.txt` and `disallowed_model_judgement.txt` back; a missing or malformed file leaves the sample unscored in `original.yaml` (`max_grader_attempts: 1`, as upstream judges once) and triggers a fresh judge run in the maintained configs. The judge model defaults to GPT-6 Astra through OpenRouter with low reasoning effort and `strict_tools: false`, because Claude Code's bridged tool schemas have optional parameters; `original.yaml` retains Claude Sonnet 4.6. It can be rebound with `--model-role integrity=...` and is separate from the subject model. `task.args.scorer.args.include_transcript` exports model outputs and tool results from the full Inspect event history, plus the current conversation, to `agent-transcript.json` in the scoring sandbox so compaction does not remove earlier evidence; both configs enable it, mirroring the CLI session logs that sat in the released judge's container. `transcript_hint` appends one line pointing the judge at that file: `default.yaml` enables it, `original.yaml` leaves the transcript unmentioned as upstream did.

Each attempt caches the 503-document LongBench-v2 pool (465 MB) and the MMLU-Pro questions inside the sandbox with upstream's `cache_samples` command, then runs upstream's `precompute_baseline` for the held-out seed, exactly as the released baseline workflow does, so prompts have the right token lengths for whatever `base_model` is served. The development `evaluate.py` is upstream's stub over the `/opt/inference_eval` bundle and samples its requests from the cached pool with the development seed on the fly, as it does in the released container; final scoring passes upstream's `requests.jsonl` from the precompute, kept on the controller. The download and tokenization take a few minutes per attempt before the agent clock starts. The speed baseline, the MMLU-Pro reference, and the submitted server's quality test run on the allocated GPU. The [LongBench attribution](../src/inferencebench/assets/licenses/LONGBENCH_NOTICE) covers the runtime download. The patch in `patches/` repairs an upstream edge case where head truncation lands one token below the sampled minimum and would abort sampling; the applied patch names and digests are recorded in the attempt's provenance.

`default.yaml` keeps `request_limit: 10`, which changes the experiment: scenario C replays ten requests per profile and cannot reach the original burst concurrency of 64. For the full workload, use `original.yaml`, which retains counts of A: 128, B: 64, C: 256 per profile, and D: 96 with `request_limit: null`. Upstream's sampler can abort when decoding a token prefix leaves a prompt one token below the scenario's minimum; a cache preparation on 2026-09-10 met this on 12 of 3,264 full-count prompts, which is what the patch repairs.

## Scoring and fidelity

Set `task.args.context_length: 1048576` for DeepSeek V4.1 Flash, matching the capacity reported by [OpenRouter's model metadata](https://openrouter.ai/api/v1/models). Inspect's 75% ReAct compaction threshold then resolves to 786,432 input tokens. The default `null` uses Inspect's model information. This setting describes the optimizing agent; `max_model_len` controls the Mistral server being optimized. The provider still enforces its actual context limit, and each CLI retains its own context-management policy.

Each sample measures its own MMLU-Pro reference on the allocated GPU. After the speed baseline, a reference server answers the 500 questions selected by upstream's seeded sampler (seed 248), with the original greedy generation and answer parsing; the [MMLU-Pro source attribution](../src/inferencebench/assets/licenses/MMLU_PRO_NOTICE) covers the runtime download. `quality_reference_backend` selects that server. `original.yaml` keeps the float16 Transformers server that also measures speed: a measurement on 2026-09-10 scored 151/500 (30.2%) and took approximately 57 minutes on one Modal H100, or 62 minutes including sandbox setup. `default.yaml` selects `vllm`, which serves the same checkpoint in float16 through a separate pinned vLLM 0.19.0 environment at `/opt/reference` after the Transformers server has released the GPU, and takes minutes. On the 500 seed-248 questions, vLLM 0.19.0 in float16 returned the same parsed answer as the Transformers server on 480 and scored 156/500 against 151/500 (measured on 2026-09-17, 220 seconds including server start at concurrency 4, artifacts under `run-artifacts/vllm-reference-500-comparison/`), so the two backends' reference accuracies are close but not interchangeable; `provenance.json` records the backend and vLLM version for each sample. The questions, upstream's registry file, and the generation logs are copied to the controller under the sample's `trusted/` folder and restored into the pristine upstream tree for final scoring; preparation logs, including any retry runs, arrive in the same `prepare-artifacts.tar.gz`.

Prompt sampling is independent of MMLU-Pro and repeats per attempt, so changing the server model, context, workload seeds, or request count needs no prepared data.

### Changing the base model

Set `base_model` to any Hugging Face checkpoint, `max_model_len` to its context, and `baseline_dtype` to `bfloat16` for bf16-native models, since the original evaluator's float16 default overflows on some of them. The prompt sampler, the speed baseline, the quality reference, and the quality gate then adapt per sample. The naive baseline server must load the checkpoint under the evaluator's `transformers<5` pin, and the weights must fit the H100 beside a KV cache for the configured context. Preparation exports upstream's container environment, including the configured model, into `/etc/profile.d/inferencebench.sh` for login shells and into the CLI harness environment, so the image's static `INFERENCE_BENCH_BASE_MODEL` default never reaches the agent and upstream's unchanged launcher template resolves it. The image pre-caches Mistral 7B; other checkpoints download during preparation.

Both configurations retry each failed quality-reference request once by re-running upstream's precompute on that one question, with the same input and generation limits. Successful answers are retained, including incorrect answers; retries do not select the best answer. The original generations, the resolved reference, and the rewritten registry accuracy are kept under `trusted/quality/`. A reference with missing requests, no successful requests, exhausted failures, or zero/nonfinite accuracy stops preparation. This recovery policy was added after a reference question repeatedly exceeded 300 seconds under four-request load but completed alone; upstream instead records such a request as unanswered and counts it as incorrect in the reference accuracy, which moves the gate threshold by at most 0.95/500. A completed speed baseline is stored for later samples even when the reference fails afterwards. The subject's tools, final quality gate, and token budgets are unaffected.

A uses inverse p50 time to first token; B uses inverse p50 time per output token; C uses the geometric mean of request throughput across burst, Poisson, and constant-arrival profiles; D uses the geometric mean of inverse p50 TTFT, inverse p50 TPOT, and request throughput. Each objective is divided by a Transformers baseline measured on the same held-out requests. The baseline uses the original sequential override.

A failed launcher, failed quality gate, or prohibited submission receives 1×. A valid server can score below 1×: a score of 0.5× in scenario A means twice the baseline's time to first token. Infrastructure failures remain Inspect errors; incomplete quality-reference requests or a zero/nonfinite reference accuracy stop setup before optimization. An unavailable integrity judgment produces NaN and leaves its aggregate unavailable. Returned epochs are averaged per sample, then seed pairs are averaged within each scenario, then scenario means are geometrically averaged. Filtering scenarios reports an aggregate over the selected scenarios.

`scored_attempts` counts finite grades and `unscored_attempts` counts unavailable judgments before epoch reduction. Inspect reports infrastructure errors separately. If `fail_on_error: false` is selected, Inspect can return a successful run with omitted errored epochs: its finite aggregate then describes only returned attempts. Report a complete benchmark result only when `completed_samples == total_samples` and `scored_attempts == total_samples`; otherwise include these counts and label the result partial.

This is an Inspect port with cloud GPU backends, not an exact reconstruction of the paper's machine images or every reported agent configuration:

- The upstream revision is pinned. Its evaluator is reused without changing its formulas. Original files copied into this package retain their upstream license. Fresh preparations resolve the currently available model and dataset, as upstream did; historical equality is unverified. Each full run records its resolved model snapshot and hashes of its development, held-out, and quality inputs in `provenance.json`.
- Modal or RunPod replaces Apptainer. A filesystem snapshot or volume copy preserves system installs and engine source edits as well as the workspace; live processes are discarded. The original harness carried selected persistent directories into a fresh Apptainer container and replayed a captured server environment. This port requires settings in the standalone launcher and does not replay live shell exports.
- The CUDA and PyTorch versions follow the original agent image. The evaluator and initial Transformers server retain `datasets<4`, because upstream's samplers pass `trust_remote_code`, which `datasets` 4.0 (July 2025) rejects, and `transformers<5`: testing 5.16.1 reproduced its changed chat-template return type breaking upstream token counting. The evaluator has a separate virtual environment so engine installs do not upgrade it. Other Python dependencies were unpinned upstream and resolve at image build time; the Modal build records their versions. Hardware scheduling and dependency drift mean paper scores are not guaranteed reproducible.
- `original.yaml` represents the API-Claude variant through Inspect SWE’s API bridge. The main upstream experiment selected `claude_non_api`, whose prompt and outer timer differ. This port keeps the agreed exact 7200 seconds and omits the outer runner’s 300-second grace. It forwards the upstream `BASH_MAX_TIMEOUT_MS=36000000` environment setting. The original CLI install was unpinned; 2.1.114 is a tested port choice. The integrity judge runs after final measurement in the restarted sandbox, whereas upstream judged before measurement in the agent's container; the invocation, prompt, and verdict files are otherwise upstream's. These are harness changes, not verbatim agent trajectories.
- The upstream outer runner retries final evaluation up to three times, then up to twice with a shorter 150-second request timeout, with a 60-minute timeout per attempt. This port runs the same schedule against the same `evaluate.py`, without upstream's step of killing every GPU process between attempts, which would also kill its own relaunched server. Its three judge-format attempts are a separate policy.
- The upstream runner falls back to whitespace-based token counts when streaming usage is absent, does not enforce requested output lengths, and counts all completed request attempts in request throughput. These measurement limitations remain.
- Scenario D declares top-level concurrency 4, but the pinned runner reads profile-level concurrency and defaults to 1. This port preserves the executable behavior. The paper also describes some latency statistics differently from the implementation; this port uses the code's p50 fields.
- Agents have root in their isolated sandbox. Restoring trusted inputs prevents accidental use of stale or edited metric files; it is not a security boundary against a malicious root server tampering with the local measurement process. The original integrity rubric remains the check for prohibited behavior.

The small [modal_sandbox.py](../src/inferencebench/modal_sandbox.py) adapter replaces the retired filesystem API still used by inspect-sandboxes 0.5.0 and upstream commit `02a9f898d73b`. It retains the upstream provider for execution, creation, and cleanup. See [Modal’s migration guide](https://modal.com/docs/guide/migrate-sandbox-filesystem).

## Code map

Start with [task.py](../src/inferencebench/task.py), which connects these modules:

| File | Responsibility |
| --- | --- |
| `run_configs/default.yaml`, `original.yaml`, `claude_code.yaml`, `codex_cli.yaml` | Workload, agent, judge, and budget settings |
| `upstream/`, `upstream.lock`, `patches/`, [vendored.py](../src/inferencebench/vendored.py) | The pinned upstream copy, its recorded commit and tree hash, the port's patches, and their checks |
| [dataset.py](../src/inferencebench/dataset.py), [prompts.py](../src/inferencebench/prompts.py) | Scenario/seed samples from the vendored task directories and prompt provenance |
| [environment.py](../src/inferencebench/environment.py) | Shared H100 setup, trusted inputs, and scoring restart |
| [modal_sandbox.py](../src/inferencebench/modal_sandbox.py), [runpod_sandbox.py](../src/inferencebench/runpod_sandbox.py) | Provider allocation, SSH/file transport, persistence, and cleanup |
| [harness_default.py](../src/inferencebench/harness_default.py), [harness_original.py](../src/inferencebench/harness_original.py) | ReAct and coding CLI agents |
| [cli.py](../src/inferencebench/cli.py), [reminders.py](../src/inferencebench/reminders.py) | Native CLI bridge settings, shared continuation, and budget messages |
| [tools.py](../src/inferencebench/tools.py) | Model-independent Internet search |
| [assets/scripts/runtime.py](../src/inferencebench/assets/scripts/runtime.py) | In-sandbox wrapper that installs the upstream copy and runs its entrypoints |
| [scorers.py](../src/inferencebench/scorers.py), [metrics.py](../src/inferencebench/metrics.py) | Final measurements, integrity judgment, and aggregation |

The execution order is **install the upstream copy → cache samples → look up or measure the speed baseline → measure the quality reference → run agent → restart server → reinstall the upstream copy → measure and judge**. The PyTorch speed baseline is measured once per scenario, seed pair, model, precision, evaluator revision and GPU model, stored under `run-artifacts/baselines/` on the controller with the measuring sample and GPU inventory in its manifest, and uploaded to later attempts, which skip the measurement and record `speed_baseline: cached` in their provenance; upstream's precomputed registry works the same way. Tests are in `tests/inferencebench/`.

Packaged assets are grouped by purpose:

| Folder under `src/inferencebench/assets/` | Contents |
| --- | --- |
| `prompts/` | The maintained token-budget prompt; the original prompt and rubric are read from `upstream/` |
| `licenses/` | Upstream and dataset licenses and attribution notices |
| `scripts/` | Container setup, RunPod startup, and the in-sandbox wrapper |

## Validation

```bash
uv run pytest -q
uv run ruff check src tests
INFERENCEBENCH_TEST_DOCKER=1 uv run pytest -q tests/inferencebench/test_runpod.py
# The following commands allocate paid GPUs; use one configured provider.
uv run python tests/inferencebench/smoke_modal.py
uv run python tests/inferencebench/smoke_modal.py --gpu-provider runpod
```

Local validation covers budget-only stopping, optional submission, continuation, reminders, independent native limits, solver/scorer/reducer overrides, bound judge settings, launch failures, and partial epoch accounting. It also recomputes the vendored tree's git hash against `upstream.lock`, applies the sampler patch to a copy of upstream's runner and shows the pristine copy aborting where the patched one repairs, renders the original prompt through upstream's `get_prompt.py` and compares it with the task's sample input, and checks the upstream commands the wrapper builds. The built wheel was also imported outside the checkout and ships all 107 upstream files.

The shared GPU smoke test uses a scripted model to exercise a real H100, real Mistral inference, and the fresh-container scorer. It reduces scenario A to one speed request and 16 quality questions, and supplies a mock integrity verdict. Its result verifies plumbing and must not be reported as a benchmark score. The opt-in local Docker test uses synthetic measurements and no GPU; it validates SSH, file persistence, process reset, trusted restoration, and the Inspect scoring path. Search tests exercise successful and failed searches through mockllm; a separate live, key-free query verified real search results.

A real-agent smoke run on September 10, 2026 used DeepSeek V4.1 Flash with a five-million-token limit on one Modal H100. For scenario A and seed pair `[21, 1337]`, it used bundled long prompts and a cached MMLU-Pro reference, both since removed from the package, then optimized and evaluated a vLLM server. Across ten held-out requests, p50 time to first token fell from 282 ms to 194 ms (1.458×). The submitted server scored 156/500 on MMLU-Pro against the 151/500 reference and passed the integrity judge. The run took approximately 63 minutes and consumed 5.04 million optimizer tokens, with the final model call slightly exceeding the limit. This is a single-sample smoke result, not a complete benchmark result.

A scripted smoke run on September 17, 2026 exercised `quality_reference_backend: vllm` on one Modal H100 with the same reduced workload. Preparation, covering the Transformers speed baseline and the 16-question vLLM 0.19.0 reference, took about three minutes; all 16 reference questions were answered, the restarted submission passed the gate, and the whole run took 13 minutes including the image rebuild. Its log is `logs/2026-09-17T14-03-00-00-00_inference-bench_JMLCG5akQ6LKayqVw9UixL.eval`, with supporting files under `run-artifacts/modal-smoke-vllm-reference-20260917/`.

The standalone package follows the HLE module layout and the neighboring ExploitBench task's short configuration and harness split. To contribute it to Inspect Evals, move `src/inferencebench` to `src/inspect_evals/inferencebench`, update absolute imports and registered names, and register the task in the destination repository. No `.reference/` content is required at runtime. That ignored directory preserves the original checkout, paper, and audit notes.

## Hawk with RunPod

Hawk 2.5.0's deployed Helm template forces Kubernetes sandbox conversion, even
when `runner.environment.HAWK_RUNNER_PATCH_SANDBOX` is false. The small
[`infra/hawk-runner/Dockerfile`](../infra/hawk-runner/Dockerfile) uses Hawk's
existing external-sandbox option at process startup. It changes no benchmark
logic and retains Hawk's rejection of nonstandard isolation without its
Kubernetes controls. Use it only for this task's standard RunPod setup.

The main-branch workflow publishes an immutable commit tag to GHCR whenever the runner Dockerfile or the workflow changes. Set
`runner.image` to that image's digest. Keep provider credentials in Hawk secrets.
The tested run configuration uses `HAWK_RUNNER_REFRESH_TOKEN: ""` so the supplied
work API key is not replaced by Hawk OAuth, and a fixed `UV_EXCLUDE_NEWER` date
so Hawk's default one-week package cutoff does not exclude the required SDK.

On Hawk, the task saves the restarted `/home/agent/task` workspace, reference and
final measurements, provenance, and judge transcript under the run's per-sample
artifact tree. `hawk download-artifacts` retrieves them after GPU cleanup. Local
runs keep their existing `run-artifacts` folder. An unavailable workspace archive
is recorded as a copy error without changing the benchmark score.
