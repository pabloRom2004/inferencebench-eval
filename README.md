# InferenceBench

[InferenceBench](https://arxiv.org/abs/2607.20468) tests agents that deploy and optimize Mistral-7B-Instruct-v0.3 inference across four workloads on one H100 80GB, measuring speedup while retaining model quality.

Contributed by [@pabloRom2004](https://github.com/pabloRom2004).

## Usage

### Installation

Requires Python 3.12+, your model provider's API key, and a Modal or RunPod account. From this checkout:

```bash
uv sync                 # ReAct and CLI support
uv run modal token new  # Authenticate the default GPU provider
```

### Running evaluations

> [!NOTE]
>
> Each attempt allocates one H100 on Modal or RunPod. GPU setup and inference incur cloud charges. Model-provider requests run on the Inspect controller; your provider API key is not sent to the GPU sandbox. See the [implementation guide](docs/implementation.md#switching-gpu-providers) for RunPod setup.

> [!NOTE]
>
> Each attempt samples its long prompts from LongBench-v2 with the original seeded sampler, measures the Transformers speed baseline and the 500-question MMLU-Pro reference on its own GPU before optimization starts, then checks the submitted server's quality after restart. Both measurements are taken once per model, workload, and GPU model and shared with every later attempt from `run-artifacts/baselines/`, as upstream's precomputed registries are; the first attempt on a fresh controller pays about an hour for the Transformers reference.

Replace `provider/model` with your Inspect model identifier. Run ReAct: **(Recommended way to run the eval)**

Review the settings in [default.yaml](src/inferencebench/run_configs/default.yaml) and adjust them as needed before running the evaluation.

```bash
uv run inspect eval \
  --run-config src/inferencebench/run_configs/default.yaml \
  --model provider/model \
  -T scenarios=A -T 'seed_pairs=[[21,1337]]' \
  --log-dir logs
```

Or select a provider CLI in the same [default.yaml](src/inferencebench/run_configs/default.yaml):

```bash
uv run inspect eval \
  --run-config src/inferencebench/run_configs/default.yaml \
  --model provider/model \
  --solver inferencebench/default_agent \
  -S harness=codex_cli \
  -S 'harness_args={"version":"0.154.0","web_search":"disabled"}' \
  -T scenarios=A -T 'seed_pairs=[[21,1337]]' \
  --log-dir logs
```

Keep `--solver inferencebench/default_agent` when passing `-S` overrides; Inspect requires an explicit solver name for these CLI arguments. Other Inspect SWE CLIs (gemini_cli, kimi_code, opencode) use the same `inferencebench/default_agent` selector with `-S harness=opencode` and an explicit `-S 'harness_args={"version":"1.18.31"}'`.

Or the original configuration:

Review the settings in [original.yaml](src/inferencebench/run_configs/original.yaml) and adjust them as needed before running the evaluation.

```bash
uv run inspect eval \
  --run-config src/inferencebench/run_configs/original.yaml \
  --model provider/model \
  -T scenarios=A -T 'seed_pairs=[[21,1337]]' \
  --log-dir logs
```

The original config gives each attempt two optimization hours and uses full request counts. It retains the original prompt and API-Claude harness variant; cloud sandboxes and caching make this a port, not an exact paper reproduction.

ReAct and `cli_agent` resume early completions until their token budget. The original wrapper resumes until its deadline. All use the same setup and final scorer.

View results with `uv run inspect view --log-dir logs`.

## Options

Edit or copy one of the configs below and pass its path to `--run-config`. Override task arguments with `-T`, harness arguments with `--solver <name> -S`, and generation/evaluation settings with CLI flags, e.g. `--token-limit 5000000 --epochs 1`. Use `uv run inspect eval --help` for all options.

To change benchmark prompt wording, edit the named `Prompt` objects in [prompts.py](src/inferencebench/prompts.py). Provider setup is in the [implementation guide](docs/implementation.md). The [fidelity review](docs/fidelity.md) traces the task, subject, judge, and differences from the paper.

## Upstream code

The authors' repository is vendored byte-for-byte at [src/inferencebench/upstream](src/inferencebench/upstream), pinned by [upstream.lock](src/inferencebench/upstream.lock) (commit and git tree hash, which a test recomputes). Every sample installs that copy into its sandbox and applies the patches in [src/inferencebench/patches](src/inferencebench/patches); each stage then runs upstream's own entrypoints: `cache_samples`, `precompute_baseline`, `precompute_quality_baseline`, the task `evaluate.py`, `create_timer.sh`, and `get_judge_prompt.py`. The Python modules here are the wrapper: sandboxes, model routing, budgets, artifact capture, and the Inspect scorer. To move to another upstream commit, clone it, copy the tree over `upstream/`, update the lock's commit and tree hash (`git rev-parse <commit>^{tree}`), and re-check the patches with `git apply --check`.

## Code layout

The package root contains the benchmark: [task.py](src/inferencebench/task.py), dataset, prompts, tools, scorers, and metrics. The two harness entrypoints are beside the task:

| File or folder | Responsibility |
| --- | --- |
| [harness_default.py](src/inferencebench/harness_default.py) | Connect task tools, reminders, and stopping rules to the selected ReAct or CLI harness. |
| [harness_original.py](src/inferencebench/harness_original.py) | Preserve the benchmark's historical agent behavior. |
| [environment.py](src/inferencebench/environment.py) | Apply task-specific sandbox configuration and setup. |
| [utils/harnesses/](src/inferencebench/utils/harnesses/) | Shared ReAct construction and CLI execution; `cli/` contains context settings, timeouts, verified downloads, bridge support, and checkpoints. |
| [utils/sandboxes/](src/inferencebench/utils/sandboxes/) | Reusable Kubernetes configuration rendering, Modal filesystem operations, and RunPod allocation, SSH/file transport, and cleanup. |
| [utils/](src/inferencebench/utils/) | Shared resource reminders, recovery bundles, and YAML loading. |
| [assets/sandboxes/](src/inferencebench/assets/sandboxes/) | Provider configuration YAML. |
| [diagnostics/](src/inferencebench/diagnostics/) | GPU-free deployment and model-provider probe. |
| [run_configs/](src/inferencebench/run_configs/) | Only `default.yaml` and `original.yaml`; harnesses and experiment variants are parameters. |

The Python files under `utils/` are byte-for-byte identical in both benchmarks and do not import task modules or prompts. The harness machinery comes from ExploitBench; the shared Modal and RunPod operations were extracted from InferenceBench's existing providers. Maintain shared fixes in ExploitBench, copy the affected files to InferenceBench, and test both repositories. Provider dependencies are needed only when using that provider; ExploitBench's development environment includes them for utility type checks (the Modal adapter requires Python 3.12 or later).

InferenceBench supplies its search tools, workspace environment, optimization deadline, and trusted GPU scoring restart. Its Dockerfile remains beside its build assets. Public task and agent names and the main run-config paths are unchanged. Sandbox YAML now lives in `assets/sandboxes/`; checkpoint keys and recovery-bundle fields are preserved.

Use `run_configs/default.yaml` for every maintained harness: set `solver.args.harness` to `react`, `codex_cli`, `claude_code`, or another supported CLI, and place native options in `harness_args`. The root `default_agent` selects the harness; the existing `react_agent` and `cli_agent` entrypoints remain available. Task setup and final scoring remain in place when the harness changes. Save experiment snapshots under `run-artifacts/<run-name>/`, keeping `run_configs/` limited to `default.yaml` and `original.yaml`.

## Parameters

Harness parameters live in `solver.args`. Shared settings are listed last.

### Default (ReAct)

Config: [default.yaml](src/inferencebench/run_configs/default.yaml) (recommended).

- `tools`: `[bash, python, web_search]`.
- `tool_timeout`: seconds allowed per shell or Python call; `7200`. Model API timeouts do not bound tool execution.
- `compaction_threshold`: context fraction triggering compaction; `0.75`.
- `token_budget_reminder`: show token usage; `true`.
- `nudge_prompt`: continue after normal completions; `true`.
- `submit`: expose a stopping tool; `false`.

### CLI

Config: [default.yaml](src/inferencebench/run_configs/default.yaml), with `solver.solver: inferencebench/default_agent` and the selected `harness`.

- `harness`: native Inspect SWE factory; `claude_code` or `codex_cli`.
- `harness_args`: native options. Verified Claude Code settings are `version: 2.1.267` with `permission_mode: bypassPermissions`, `retry_refusals: 0`, and `env: {BASH_MAX_TIMEOUT_MS: "36000000"}`; verified Codex settings are `version: 0.154.0` with `web_search: disabled`, because its native search runs on OpenAI's side and cannot follow the bridge.
- `cli_poll_timeout`: seconds a remote-process poll may wait; `7200`. Native adapters otherwise leave this unset.
- `nudge_prompt`, `token_budget_reminder`, `web_search_args`: same defaults as ReAct.

The adapter supplies the task directory, root sandbox user, bridged search, and served-model context and output limits; OpenCode also gets native compaction settings and provider timers matching Inspect's attempt timeout, and the pinned Codex release archive is staged from its published digest so concurrent samples do not hit GitHub's API limits. Native multi-attempt scoring is disabled because final grading restarts the server. Direct `--solver inspect_swe/...` overrides remain supported but omit these shared policies.

### Original

Config: [original.yaml](src/inferencebench/run_configs/original.yaml).

- `harness`, `version`: `claude_code`, `2.1.114`.
- `continue_until_deadline`: resume early exits; `true`.
- `agent_seconds` (`task.args`): optimization deadline; `7200`, starting after preparation.

Generation and evaluation settings are listed below.

### Shared

Defaults apply to ReAct/CLI unless marked otherwise. Task settings live in `task.args`.

- `scenarios`: ID or list; `A`, original `null` selects A–D.
- `seed_pairs`: development/evaluation seeds; `[[21,1337]]`, original three pairs.
- `gpu_provider`: `modal` or `runpod`; `modal`.
- `base_model`: `mistralai/Mistral-7B-Instruct-v0.3`. Any Hugging Face checkpoint the evaluator's Transformers stack can load; see the [implementation guide](docs/implementation.md#changing-the-base-model).
- `baseline_dtype`: precision of the Transformers speed baseline and the reference server; `float16` (upstream). Use `bfloat16` for bf16-native models.
- `context_length`: optimizing model's context window; `null` uses Inspect's metadata. Set `1048576` for DeepSeek V4.1 Flash.
- `strict_prompt`: insert the leaderboard's strict rules (no third-party pre-quantized checkpoints, no modifying the evaluation harness) after the base-model constraint; `true`. The paper's Table 2 used the plain prompt; the site's dagger-marked rows used a strict prompt whose text is unreleased, so the wording is the port's.
- `automated_tuning`: append an explicit instruction to use an automated hyperparameter search tool or programmatic search loop instead of manually selecting trials; `false` in both configs. Enable with `-T automated_tuning=true`.
- `seeded_arrivals`: seed scenario C's Poisson arrival times from the requests' LongBench seed (development seed for the agent's `evaluate.py`, held-out seed for the baseline and final scoring); `true` by default, `false` in the original config, which keeps upstream's unseeded draw.
- `scenario_a_output_tokens`: cap Scenario A's forced output length for the agent's `evaluate.py`, the speed baseline, and final scoring. Scenario A scores median TTFT at concurrency 1, so decode tokens only cost time; `16` by default (a few tokens so the first streamed chunk always carries text), `null` in the original config, which keeps upstream's 819 to 1024.
- `agent_seconds`: optimization wall-clock limit, starting after preparation; `36000` (10 hours) by default, `7200` in the original config. The maintained prompt reflects the deadline; `-T agent_seconds=null` removes it. Preparation and final scoring take additional time.
- RunPod's [provider configuration](src/inferencebench/assets/sandboxes/runpod.yaml) separately allows `7200` seconds for saving the filesystem before grading (`snapshot_timeout_seconds`) and `7200` seconds for initial startup or restoration (`startup_timeout_seconds`). These infrastructure allowances do not extend `agent_seconds`; customize them with `gpu_config`.
- `request_limit`: requests per profile; `10`, original `null` preserves full counts.
- `quality_samples`, `quality_seed`: `500`, `248`.
- `quality_reference_backend`: server measuring the MMLU-Pro reference; `transformers` in both configs, the original naive server. `vllm` (pinned 0.19.0) answers in minutes but scores a few questions higher on the same weights, which tightens the gate. Upstream's precompute runs the Transformers reference at concurrency 1 and other backends at `quality_concurrency`; the port does the same. The reference is measured once per model, backend, question selection, precision, upstream commit and GPU model, shared from `run-artifacts/baselines/`, and the same entry serves both configs.
- `quality_baseline_max_attempts`: `2` retries a failed reference request once on its own and requires a complete reference; `1` accepts upstream's registry exactly as its precompute wrote it.
- `quality_tau`: required fraction of reference accuracy; `0.95`.

Generation (`generate_config`):

- `temperature`: `null` (provider default).
- `max_tokens`: output cap per response; `null` (provider/Inspect default).
- `seed`: `null` (unset).
- `reasoning_effort`: `null` (provider default).
- `max_retries`: API retries; `10`.
- `attempt_timeout`: API attempt timeout; `900` seconds.

Evaluation (`eval_config`):

- `limit`: selected samples before epoch repetition; `null` means all selected.
- `epochs`: attempts per scenario/seed pair; `1`.
- `token_limit`: cumulative input-plus-output tokens, including cached input; `100000000`, original `null`.
- `time_limit`: native attempt deadline; `null`. The original uses `agent_seconds` above.
- `score_on_error`: `true` in both profiles. A solver error still sends its partial submission through the normal restart, quality, speed, and integrity checks; the original error remains in the log. Infrastructure failures in grading remain errors, with no invented score. Hawk retains the submission and transcript before the scoring restart as well as after it, so a failed filesystem snapshot does not discard the task directory.
- `max_samples`: concurrent attempts; `1`.

The separate integrity judge runs the way upstream's does: upstream's `get_judge_prompt.py` renders the rubric with the restarted launcher and server-log tail, Claude Code (`judge_cli_version`, run through Inspect SWE with the model routed by Inspect) executes in the submission directory with permissions bypassed, and the two verdict files it writes decide the outcome. The judge model is GPT-6 Astra by default and Claude Sonnet 4.6 in the original config; change `model_roles.integrity` to replace it. Both configs export the agent transcript into the scoring sandbox, matching the released container where the CLI session logs sat unreferenced; only the default config's prompt points the judge at it. The original config judges once (`max_grader_attempts: 1`); the maintained configs re-run the judge when its verdict files are missing or malformed.

## Dataset

Four scenarios, read from the vendored [task directories](src/inferencebench/upstream/src/eval/tasks), with one seed pair by default and three in the original configuration. The default selects A; all scenarios give four or twelve samples respectively. Original request counts are A: 128, B: 64, C: 256 per profile, D: 96.

[Assets](src/inferencebench/assets) contain the maintained token-budget prompt, scripts, and licenses; the original prompt and judge rubric are read from the vendored upstream copy. The LongBench-v2 prompts are sampled inside every attempt with upstream's sampler; the MMLU-Pro reference is measured once per model and GPU model and shared. Transformers measurements recorded 151/500 correct (2026-09-10 and 2026-09-17); vLLM 0.19.0 recorded 156 and 154 on the same questions.

## Scoring

![InferenceBench flow: Prompt → Agent → Grader, with development feedback, scenario objectives, sampled prompts, and final quality and integrity checks.](docs/grading-flow.svg)

The agent edits `start_server.sh` and tests with `evaluate.py`. Final scoring restarts the saved submission, measures held-out speed and MMLU-Pro accuracy, and judges integrity. **speedup** divides the scenario objective by the Transformers baseline for that scenario, seed pair and GPU model, measured by the first attempt that needs it and shared with later attempts; the quality gate compares against an MMLU-Pro reference shared the same way. **aggregate_speedup** averages epochs, then seeds within scenarios, and geometrically averages scenario means.

Invalid submissions receive 1×; valid slowdowns can score below 1×. Unavailable integrity judgments remain unscored; infrastructure failures remain Inspect errors. Report incomplete runs with their completed, scored, and unscored attempt counts.

## Changelog

### [15] - 2026-09-21

- Vendor the pinned upstream repository byte-for-byte under `src/inferencebench/upstream/` with `upstream.lock` (commit and git tree hash, checked by a test) and move the sampler's one-token truncation repair into `patches/`; every sample installs that copy and applies the patches inside its sandbox, and the shared speed baseline is keyed by the upstream commit and patch digest.
- Drive each stage through upstream's own entrypoints instead of its private functions: `cache_samples`, `precompute_baseline` (with upstream's torch settings, a 900-second request timeout and sequential requests), `precompute_quality_baseline` (concurrency 1 for the Transformers reference, `quality_concurrency` for vLLM), the task's `evaluate.py` with upstream's three-plus-two retry schedule, `create_timer.sh`, and `get_judge_prompt.py` for the judge's rubric and pre-loaded evidence.
- Install the agent workspace as upstream's harness does: the unchanged launcher template, upstream's `evaluate.py` stub over the `/opt/inference_eval` bundle, the `/opt/inference_eval/baselines` caches and registries, and the original container environment (`INFERENCE_BENCH_*`, `HOST`, `PORT`, `NUM_HOURS`) for login shells and CLI processes; the development evaluator samples its requests from the cached pool on the fly, as upstream's does.
- Render the prompt exactly as upstream's `get_prompt.py` (integer hours, `metrics_preview.json`, no trailing newline), verified by a test that runs the script.
- `quality_baseline_max_attempts: 1` now accepts upstream's registry as measured; `2` keeps the isolated retry and completeness requirement.
- Select ReAct or a native CLI through `default.yaml`, with native options in `harness_args`, `cli_poll_timeout`, OpenCode native compaction and provider timers, staged Codex release archives, and a ReAct `tool_timeout`.
- Judge with upstream's own invocation: Claude Code (`judge_cli_version`) in the restarted submission directory with the prompt rendered by `get_judge_prompt.py`, reading the two verdict files it writes. The hand-built inspection and shell tools, adapter prompts, and the `preload_evidence` and `judge_shell` toggles are removed; `original.yaml` judges once like upstream.
- Share the MMLU-Pro reference like the speed baseline: measured once per model, backend, question selection, precision, upstream commit and GPU model, stored under `run-artifacts/baselines/quality-*`, uploaded to later attempts with the exact questions it was measured on, and keyed independently of the configuration name so default and original share one entry. Default returns to the Transformers reference: on the same 500 questions vLLM scored 3 to 5 more correct answers than Transformers, enough to flip a submission at the 0.95 gate.

### [14] - 2026-09-17

- Measure the naive PyTorch speed baseline once per scenario, seed pair and GPU model and share it with later attempts from `run-artifacts/baselines/`, matching upstream's precomputed baselines.
- Restore upstream's one-token truncation repair inside the sandbox sampler, recorded per attempt in provenance, so full-count prompts (scenario A seed 999, B and C on several seeds) no longer abort preparation.
- Scale RunPod SFTP transfer time with payload size, so CLI bundle uploads such as OpenCode's no longer hit the 30-second API timeout, and end the agent budget cleanly when a transfer is cut off at the deadline.
- Wait out RunPod capacity shortfalls, reported by its API as HTTP 500 on pod creation, with spaced creation retries (`create_retry_attempts`, `create_retry_interval_seconds`) instead of failing the sample, and record the server's error text in `pod.json`.
- Retry a timed-out quality-reference request once in `original.yaml` as well (`quality_baseline_max_attempts: 2`), keep a completed speed baseline when the reference fails afterwards, and report a released RunPod pod as unavailable instead of failing the whole evaluation when a sample retry queries its SSH connection.
- Pin `datasets<4` in the sandbox image; upstream's MMLU-Pro and LongBench samplers pass `trust_remote_code`, which newer releases reject.
- Remove the bundled MMLU-Pro reference cache and its preparation CLI; every attempt measures the quality reference on its own GPU.
- Add `quality_reference_backend`: the default measures the reference with a pinned vLLM 0.19.0 server; `original.yaml` keeps the Transformers server.
- Give both configs the released judge's inline evidence and shell access (`preload_evidence`, `judge_shell`) and export the transcript in both; `transcript_hint` mentions it only in the default prompt.
- Add `strict_prompt` (default `true` in both configs), inserting the leaderboard's strict rules after the base-model constraint.
- Add `baseline_dtype`, and write the configured model into the launcher fallback, login shells, and CLI environment so no image default names Mistral.
- Remove the bundled LongBench request caches and their builder; every attempt downloads LongBench-v2 and runs the original sampler and truncation for its own tokenizer.

### [13] - 2026-09-12

- Standardize CLI adapters and continuation in `harness_default.py` and `harness_default.py`, following ExploitBench.
- Preserve model and tool evidence for the judge after context compaction and retain the full upstream scenario description.
- Prevent background output from corrupting RunPod tool completion records.
- Trace the paper and released implementation, including cache provenance and remaining differences.

### [12] - 2026-09-10

- Add `context_length` to configure the optimizing model's context metadata for compaction and bridges.

### [11] - 2026-09-10

- Bundle full long-prompt caches for the original configuration, preserving all counts and seeds; record 12 one-token truncation corrections.

### [10] - 2026-09-10

- Bundle the 500-question Mistral reference at seed 248, with per-question `.eval` logs, JSON summaries, and standalone preparation for other settings.
- Organize assets by purpose. A DeepSeek V4.1 Flash smoke run achieved 1.458× on one Scenario A sample; this is not a complete benchmark result.
