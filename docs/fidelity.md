# Fidelity review

Reviewed against the [paper](https://arxiv.org/abs/2607.20468) and released
[implementation at `24cdf88`](https://github.com/aisa-group/InferenceBench/tree/24cdf88f6a4e14ed85d665aa132cecccb3ee95ef).
The repository is vendored byte-for-byte under `src/inferencebench/upstream/`;
a test recomputes its git tree hash against `upstream.lock`. Each sample installs
that copy in its sandbox, applies the two patches in `patches/` (the sampler's
boundary-token repair, and optional seeding of scenario C's Poisson arrivals,
which upstream draws unseeded; `seeded_arrivals` enables it in `default.yaml`
and leaves `original.yaml` unseeded), and runs upstream's own commands for every measured
stage. The subject prompt, judge rubric, launcher template, evaluator stub, and
scenario definitions are read from that copy rather than re-typed.

## Data flow

1. **Select the experiment.** `task.py` reads YAML settings and `dataset.py`
   makes one sample per scenario and development/evaluation seed pair. No
   prepared inputs exist; the sampler runs per attempt. `default.yaml` uses ten requests;
   `original.yaml` retains A: 128, B: 64, C: 256 per traffic profile, D: 96.
2. **Prepare one H100.** `environment.py` provisions the configured backend,
   installs the pinned evaluator, downloads Mistral-7B-Instruct-v0.3, and
   records the resolved checkpoint and input hashes. Mistral is the server
   being optimized; GLM is the agent doing the optimization.
3. **Measure the reference.** Upstream's `cache_samples` command caches the
   question pools, then `precompute_baseline` measures the Transformers server
   on this run's GPU with upstream's torch settings (sequential requests, a
   900-second request timeout). `precompute_quality_baseline` then answers the
   500-question MMLU-Pro reference at concurrency 1 on Transformers
   (`original.yaml`) or at `quality_concurrency` on pinned vLLM
   (`default.yaml`), writing the registry file upstream's gate reads. This
   happens before the optimization budget starts.
4. **Ask the subject to build a server.** Render the original placeholders
   exactly as `get_prompt.py` does (a test compares the two). The token-budget
   prompt changes the original time instructions. The agent gets root, Internet,
   engine installation, upstream's `evaluate.py` stub over the
   `/opt/inference_eval` bundle, upstream's timer, the original container
   environment, and the unchanged empty `start_server.sh`.
   It must serve the supplied checkpoint locally through the required OpenAI
   endpoints, improve the scenario metric, and retain quality.
5. **Run the selected harness.** `harness_default.py` runs native Inspect ReAct;
   `harness_default.py` runs native Inspect SWE CLIs. Both share reminders and continuation
   through `harness_default.py`. Claude Code runs inside the GPU sandbox and calls
   GLM through Inspect's bridge. The provider key stays with the controller.
   Claude Code reads task and continuation instructions from files, preserving
   their text while keeping server names out of its process arguments. This
   adds an initial file read compared with passing the prompt directly: the
   latter let an agent's `pkill -f start_server` kill its own CLI.
   `harness_original.py` separately retains the original time-based CLI loop.
6. **Restart and measure the submission.** Preserve installed files, discard live
   processes, reinstall the pristine upstream copy and the measured inputs from
   the controller, and invoke the saved standalone launcher under upstream's
   supervisor. Upstream's task `evaluate.py` then runs with its own retry
   schedule, measuring speed on the precomputed requests and asking the
   submitted Mistral server all 500 quality questions with the original greedy
   generation and answer parsing.
7. **Ask the integrity judge to inspect evidence.** Upstream's
   `get_judge_prompt.py` renders the unchanged rubric and inlines the restarted
   launcher and server-log tail; Claude Code, with the role-bound judge model
   behind Inspect's bridge, runs it in the submission directory and writes the
   two verdict files, as upstream's `claude --print` call does. Default runs additionally expose all recorded model
   outputs and tool results, including history removed by context compaction.
   Native CLI tool results are recovered from model inputs and saved once even
   when later API calls repeat the same history.
   Hawk also retains unfinished work and the transcript if the solver fails;
   such attempts remain errors and receive no success score.
   The judge returns the original two verdicts; it does not calculate accuracy
   or speed. The adapter uses the full upstream scenario name in its prompt.
8. **Produce the score.** Divide the submission objective by its same-run
   reference. A uses inverse median TTFT; B inverse median TPOT; C the geometric
   mean of request throughput across three traffic profiles; D the geometric
   mean of inverse median TTFT, inverse median TPOT, and request throughput.
   Invalid submissions receive 1×, genuine slowdowns can fall below 1×,
   unavailable judgments remain unscored, and infrastructure failures are errors.
   Aggregate by averaging epochs and seeds within scenarios, then geometrically
   averaging scenario means. Clean up only the run's own GPU resources.

## Remaining differences

These are reasons to call the result an Inspect port, rather than a reproduction
of the paper's reported model/scaffold experiment.

| Area | Released implementation and this port |
| --- | --- |
| Subject and budget | The requested GLM-5.3 Flash comparison uses ReAct and Claude Code with a billion-token budget. The original uses two hours and its own model/scaffold pairings. |
| Config structure | The main modules follow ExploitBench. InferenceBench retains its supported native `solver` YAML block and `--solver` overrides; task setup and final scoring stay independent of that choice. |
| Judge | Original: Sonnet 4.6 inside Claude Code, before final evaluation, writing verdict files. Both configs now run Claude Code (pinned by `judge_cli_version`) in the restarted submission directory with the prompt from upstream's `get_judge_prompt.py` and read the same verdict files; the model behind it is Sonnet 4.6 in `original.yaml` and GPT-6 Astra in `default.yaml`, and the judge runs after measurement rather than before. |
| Judge evidence | Upstream inlines `start_server.sh` and the last 200 lines of `server.log` into the prompt and gives the judge a shell in the agent's home; nothing points it at the agent transcript, and the released runs' vLLM model line sits inside that 200-line tail in only 61 of 217 logs. Both configs build the prompt with upstream's `get_judge_prompt.py` from the restarted submission's launcher and `final-server.log`, hand it to Claude Code unchanged, and export the Inspect transcript to `/tmp/inferencebench/agent-transcript.json`, present but unmentioned as the CLI session logs were in the released container. Verdict files an agent pre-writes are deleted before the judge runs; upstream would read them. `default.yaml` adds one line pointing the judge at that transcript (`transcript_hint`). |
| Isolation | Hawk/RunPod replaces the original scheduler/Apptainer arrangement. The full filesystem survives restart, rather than selected persistent directories. Live shell exports must be written into the standalone launcher. |
| Hardware/software | H100 80GB, minimum 16 vCPUs/180GB RAM, Ubuntu 22.04 and CUDA 12.8 follow upstream. Storage, GPU host scheduling, driver, and unpinned dependencies can differ; record them with the run. |
| Speed baseline | Upstream precomputes the PyTorch baseline once per scenario and seed pair and reuses it for every agent. The port does the same, keyed additionally by GPU model, precision, and evaluator revision, with the measuring sample recorded in the stored manifest. |
| Strict prompt | The site's dagger-marked runs used an unreleased stricter prompt that names third-party pre-quantized checkpoints and harness edits as disallowed. Both configs insert two bullets stating those rules after the base-model constraint (`strict_prompt: true`); the wording is the port's. Set it false to reproduce the paper's Table 2 prompt. |
| Inputs | Each attempt caches LongBench-v2 and MMLU-Pro with upstream's `cache_samples` and samples with upstream's tokenizer-based sampler, patched only to re-truncate when decoding drops a boundary token (upstream aborts there on some full-count seeds). Upstream's `precache_seeds.sh` writes every seed's MMLU-Pro selection to the quality-seed path, so which selection its runs used depends on shell iteration order; the port caches the quality seed explicitly. The MMLU-Pro reference is measured per attempt. |
| Reference load | Upstream's precompute measures the Transformers speed baseline with a 900-second request timeout and its Transformers quality reference at concurrency 1; earlier port versions used 300 seconds and concurrency 4, which produced timeouts. Both now follow upstream, and both measurements are shared across samples like upstream's registries. |
| Retries | Upstream retries whole final evaluations three times, then twice with a 150-second request timeout; the port runs that schedule, minus upstream's step of killing every GPU process between attempts (which would also kill its relaunched server). Both configs retry a failed reference question once by re-running upstream's precompute for that question; `quality_baseline_max_attempts: 1` accepts upstream's registry as measured. |

The paper and released code also disagree in several places. The paper describes
a bfloat16 baseline; the code defaults to float16, which the port uses. Scenario
D declares concurrency four, but the runner reads profile-level concurrency and
executes at one. The port retains the released p50 objective fields and fixed
quality seed 248. These should not be silently changed under the label of fidelity.
See the [runner](https://github.com/aisa-group/InferenceBench/blob/24cdf88f6a4e14ed85d665aa132cecccb3ee95ef/src/eval/inference/runner.py),
[reference server](https://github.com/aisa-group/InferenceBench/blob/24cdf88f6a4e14ed85d665aa132cecccb3ee95ef/src/eval/inference/servers/transformers_openai_server.py),
and [environment defaults](https://github.com/aisa-group/InferenceBench/blob/24cdf88f6a4e14ed85d665aa132cecccb3ee95ef/src/commit_utils/set_env_vars.sh).

## MMLU-Pro reference

The reference is measured once per model, backend, question selection,
precision, upstream commit and GPU model, and shared with every later sample
from `run-artifacts/baselines/`, as upstream's precomputed registry is shared
across its runs; both configurations resolve to the same entry. Both use the
float16 Transformers server (`quality_reference_backend: transformers`), which
answers the fixed 500 questions (seed 248) in about 57 minutes on an H100 and
scored 151/500 on 2026-09-10 and 2026-09-17, so the 95% gate requires 144
correct from a submission. The pinned vLLM 0.19.0 server answers them in under
four minutes but scored 156 and 154 on the same questions (480 identical parsed
answers): float16 kernel numerics flip a few near-tied greedy choices, a
difference inside the sampling noise of 500 questions but comparable to the
gate's margin, so vLLM is available and not the default. A reference request
that times out is retried once on its own in both configurations; upstream
counts it as incorrect instead, a difference of at most one question in the
gate threshold. Each optimized server still answers every question; no agent or
submission answers are reused. Speed is also measured against a shared baseline
for the same GPU model.

The fixed questions are not a secret test set against a root-capable agent.
Trusted restoration protects the evaluator from stale edits; the integrity judge
checks prohibited adaptation or cached-output serving. This limitation also
exists in the original filesystem-based task.
