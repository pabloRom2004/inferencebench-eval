# Fidelity review

Reviewed against the [paper](https://arxiv.org/abs/2607.20468) and released
[implementation at `24cdf88`](https://github.com/aisa-group/InferenceBench/tree/24cdf88f6a4e14ed85d665aa132cecccb3ee95ef).
The original subject prompt, judge rubric, and all four scenario definitions
match the packaged copies exactly. The GPU setup downloads that pinned source;
the port calls its evaluator instead of implementing another one.

## Data flow

1. **Select the experiment.** `task.py` reads YAML settings and `dataset.py`
   makes one sample per scenario and development/evaluation seed pair. Validate
   cached inputs before allocating a GPU. `default.yaml` uses ten requests;
   `original.yaml` retains A: 128, B: 64, C: 256 per traffic profile, D: 96.
2. **Prepare one H100.** `environment.py` provisions the configured backend,
   installs the pinned evaluator, downloads Mistral-7B-Instruct-v0.3, and
   records the resolved checkpoint and input hashes. Mistral is the server
   being optimized; GLM is the agent doing the optimization.
3. **Measure the reference.** The upstream Transformers server processes the
   evaluation request set on this run's GPU, with upstream's sequential baseline
   override. A reference server then answers the 500-question MMLU-Pro
   reference: Transformers in `original.yaml`, pinned vLLM in `default.yaml`.
   This happens before the optimization budget starts.
4. **Ask the subject to build a server.** Render the original model, scenario,
   mission, and endpoint placeholders. The token-budget prompt changes the
   original time instructions. The agent gets root, Internet, engine installation,
   development requests, `evaluate.py`, and an initially empty `start_server.sh`.
   It must serve the supplied checkpoint locally through the required OpenAI
   endpoints, improve the scenario metric, and retain quality.
5. **Run the selected harness.** `harness_default.py` runs native Inspect ReAct;
   `cli.py` runs native Inspect SWE CLIs. Both share reminders and continuation
   through `reminders.py`. Claude Code runs inside the GPU sandbox and calls
   GLM through Inspect's bridge. The provider key stays with the controller.
   Claude Code reads task and continuation instructions from files, preserving
   their text while keeping server names out of its process arguments. This
   adds an initial file read compared with passing the prompt directly: the
   latter let an agent's `pkill -f start_server` kill its own CLI.
   `harness_original.py` separately retains the original time-based CLI loop.
6. **Restart and measure the submission.** Preserve installed files, discard live
   processes, restore trusted evaluator inputs from the controller, and invoke
   the saved standalone launcher. The upstream evaluator measures speed on the
   evaluation requests and asks the submitted Mistral server all 500 quality
   questions with the original greedy generation and answer parsing.
7. **Ask the integrity judge to inspect evidence.** The unchanged rubric checks
   contamination, model substitution, API offloading, and other prohibited
   behavior. A separate role-bound model reads the launcher, source, and logs
   through a read-only tool. Default runs additionally expose all recorded model
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
| Judge | Original: Sonnet 4.6 inside Claude Code, before final evaluation, writing verdict files. Maintained default: GPT-6 Astra inside Inspect ReAct, after performance/quality pass, returning verdicts. `original.yaml` selects Sonnet inside Inspect ReAct with the same rubric, the same inline evidence, and a shell tool; the judge still returns verdict lines and runs after measurement. |
| Judge evidence | Upstream inlines `start_server.sh` and the last 200 lines of `server.log` into the prompt and gives the judge a shell in the agent's home; nothing points it at the agent transcript, and the released runs' vLLM model line sits inside that 200-line tail in only 61 of 217 logs. Both configs reproduce the inline block verbatim from the restarted submission's launcher and `final-server.log` (`preload_evidence`), add shell access (`judge_shell`), append only a minimal adapter that names the directory and the verdict format, and export the Inspect transcript to `/tmp/inferencebench/agent-transcript.json`, present but unmentioned as the CLI session logs were in the released container; a byte comparison on 2026-09-17 found the released prompt to be an exact prefix of the original-config prompt. `default.yaml` adds one line pointing the judge at that transcript (`transcript_hint`). |
| Isolation | Hawk/RunPod replaces the original scheduler/Apptainer arrangement. The full filesystem survives restart, rather than selected persistent directories. Live shell exports must be written into the standalone launcher. |
| Hardware/software | H100 80GB, minimum 16 vCPUs/180GB RAM, Ubuntu 22.04 and CUDA 12.8 follow upstream. Storage, GPU host scheduling, driver, and unpinned dependencies can differ; record them with the run. |
| Input caches | Frozen requests avoid repeated sampling. Twelve original full-workload prompts needed a recorded one-token truncation repair to satisfy upstream's own bounds. Seeds, selected documents, and output budgets are retained. |
| Retries | Upstream retries whole final evaluations, sometimes with shorter request timeouts. This port evaluates once and reports failures. Default reference preparation retries failed questions without resampling or selecting a better answer; original permits one attempt. |

The paper and released code also disagree in several places. The paper describes
a bfloat16 baseline; the code defaults to float16, which the port uses. Scenario
D declares concurrency four, but the runner reads profile-level concurrency and
executes at one. The port retains the released p50 objective fields and fixed
quality seed 248. These should not be silently changed under the label of fidelity.
See the [runner](https://github.com/aisa-group/InferenceBench/blob/24cdf88f6a4e14ed85d665aa132cecccb3ee95ef/src/eval/inference/runner.py),
[reference server](https://github.com/aisa-group/InferenceBench/blob/24cdf88f6a4e14ed85d665aa132cecccb3ee95ef/src/eval/inference/servers/transformers_openai_server.py),
and [environment defaults](https://github.com/aisa-group/InferenceBench/blob/24cdf88f6a4e14ed85d665aa132cecccb3ee95ef/src/commit_utils/set_env_vars.sh).

## MMLU-Pro reference

Each sample measures the reference on its own GPU before optimization. With
`quality_reference_backend: transformers` (`original.yaml`), the float16
Transformers server answers the fixed 500 questions (seed 248), about 57
minutes on an H100. With `vllm` (`default.yaml`), a pinned vLLM 0.19.0 server
answers them in float16 in under four minutes; on the same 500 questions it
matched the Transformers answer on 480 and scored 156/500 against 151/500, so
the two backends' reference accuracies are close but should not be mixed. Upstream precomputes and reuses this registry across
runs; this port repeats the measurement per sample so no bundled answers need
maintaining or checksum matching. The 2026-09-10 measurement was 151/500
(30.2%), so its 95% gate required at least 144/500 from the submission; each
run's own reference sets its gate. Each optimized server still answers every
question; no agent or submission answers are reused. Speed is also remeasured
against this run's GPU baseline.

The fixed questions are not a secret test set against a root-capable agent.
Trusted restoration protects the evaluator from stale edits; the integrity judge
checks prohibited adaptation or cached-output serving. This limitation also
exists in the original filesystem-based task.
