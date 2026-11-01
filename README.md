---
title: ML Training Run Debugger
emoji: "🧪"
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - fastapi
  - debugging
  - machine-learning
pinned: false
---

# ML Training Run Debugger

[![CI](https://github.com/geethika-sandireddy/ml-agent-debug-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/geethika-sandireddy/ml-agent-debug-benchmark/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Live Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Space-blue)](https://geetss-ml-debug-env.hf.space)

OpenEnv-compatible RL environment for evaluating whether an LLM agent can debug ML systems: it reads logs, metrics, and configs, then proposes and applies fixes, on three deterministic tasks of increasing difficulty. Packaged as a stateless-or-stateful FastAPI server so any HTTP client - human, script, or agent - can drive it, and deployed live on Hugging Face Spaces.

**Live Space:** [geetss-ml-debug-env.hf.space](https://geetss-ml-debug-env.hf.space)

## v2: real PyTorch workloads

In v1 the agent was shown scripted logs. In v2 (`mode: "real"`) every task is a real PyTorch
training script with one injected bug. The environment runs it, shows the agent the real
output, lets it read and edit the code and rerun, then grades whether the bug was actually
fixed. The v1 simulator is still available as the default `legacy` mode.

| Task | Symptom | Bug | Fix |
|---|---|---|---|
| double_softmax | slow learning, low plateau | softmax before `CrossEntropyLoss` | remove the softmax |
| norm_mismatch | validation far below training | validation inputs not normalized | normalize validation (and prediction) inputs |
| leakage | validation looks too good vs held-out data | validation reuses training rows | disjoint validation rows |
| scheduler_bug | learning rate decays to ~0, loss flatlines | `scheduler.step()` every batch | step once per epoch |
| missing_eval | noisy, lower validation accuracy | `model.eval()` never called, dropout stays on | call `model.eval()` |

Agents see neutral ids (`task_1` to `task_5`). Actions: `read_file`, `edit_file`,
`run_training`, `inspect_metrics`, `submit_diagnosis`. Episodes are capped at 15 steps and 3 runs.

**Reward** (deterministic): 0.3 for the correct diagnosis label, 0.6 for a verified fix, up to 0.1
efficiency bonus when fixed. A fix is verified by rerunning the final `train.py` in a clean copy
with a pristine helper and checking accuracy on a hidden test split (and zero train/validation
overlap for `leakage`). Editing the protected helper, or using forbidden names or imports
(for example reading the hidden data), scores 0.

**Measured evidence** (`tests/test_workloads.py`, 5 seeds): for every bug the buggy script is worse
than the fixed one, and the pass thresholds in `workloads/*/config.json` sit between the best buggy
run and the worst fixed run on hidden-test accuracy. For `leakage` the measured effect is inflated
validation accuracy and non-zero overlap; hidden accuracy is unchanged by design.

**Baselines** (`python scripts/run_eval.py`, 1 seed per task):

| agent | double_softmax | leakage | missing_eval | norm_mismatch | scheduler_bug | mean |
|---|---|---|---|---|---|---|
| noop | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| oracle | 0.98 | 0.98 | 0.98 | 0.97 | 0.97 | 0.98 |
| llm | not run | not run | not run | not run | not run | not run |

The no-op agent is the floor and the scripted oracle (applies the known fix) is the ceiling.
The LLM agent was **not run** because no API key was available when these results were produced.
Set `OPENAI_API_KEY` (and optionally `OPENAI_BASE_URL`, `MODEL_NAME`) and rerun to fill that row.

**Reproduce**: `pip install -r requirements.txt pytest && pytest` (the workload tests take several
minutes because each run starts PyTorch), then `python scripts/run_eval.py --seeds 3`.

**v2 limitations**
- The sandbox is a timeout plus CPU and heap limits, not a security boundary. Limits apply on
  Linux and macOS only; on Windows only the timeout applies.
- The source guard in the grader is a benchmark guardrail, not a proof against a determined attacker.
- Baseline numbers use 1 seed per task; thresholds were set from 5 measured seeds.
- Each workload is a tiny MLP on scikit-learn digits, so the tasks are small by design.
- The Hugging Face Space has not been updated or checked for torch memory use on its free tier.

## Architecture (v1 legacy simulator)

```mermaid
graph TD
    A[Agent / Client<br><small>HTTP: reset, step, state, grader, tasks</small>] --> B(FastAPI main.py<br>Session Resolution<br><small>header / cookie / shared default</small>)
    B --> C(MLDebugEnv env/environment.py<br>Action ➔ step<br>➔ Observation + Reward)
    C --> D(simulator.py<br><small>Scripted logs/metrics/configs per task</small>)
    C --> E(graders.py<br><small>Deterministic 0.0-1.0 score from evidence, causes, fixes</small>)
```


`inference.py` is a separate client that drives the same environment through an OpenAI-compatible LLM (or the scripted baseline in `env/baseline.py` as a fallback), emitting the `[START]/[STEP]/[END]` log contract required for scoring.

## Motivation

This benchmark targets a real engineering workflow: diagnosing failed or misleading ML training runs from incomplete evidence. The tasks reflect common debugging patterns that matter for agent evaluation:

- evidence gathering across logs, metrics, and configs
- causal reasoning from multiple weak signals
- distinguishing root causes from tempting distractors
- proposing and applying targeted fixes instead of guessing

The scenarios focus on recurring real issues: train/val preprocessing mismatch, redundant softmax before cross-entropy, and apparently healthy train/val metrics that hide leakage, imbalance, or a misleading primary metric.

## Why This Is A Useful Agent Benchmark

Many agent benchmarks test navigation or form-filling. This one tests something closer to senior ML debugging work:

- the agent must inspect multiple information sources before acting
- partial progress matters, not just final success
- wrong fixes and premature fixes are penalized
- the hard task requires uncovering multiple interacting causes

That makes the environment suitable for evaluating whether an agent can debug ML systems rather than just execute a fixed script.

## Tasks

### Task 1 - easy: train/val scaling mismatch

- Symptom: high train accuracy, flat validation.
- Likely cause: validation pipeline missing normalization that training has.
- Goal: find it in the configs, propose the matching fix, apply it.

### Task 2 - medium: double softmax

- Symptom: loss goes down while accuracy goes down.
- Likely cause: logits passed through softmax twice on the way to cross-entropy.
- Goal: trace the loss path and apply the fix.

### Task 3 - hard: silent generalization failure

- Symptom: strong train/val, weak test, no crashes.
- Causes: split leakage, class skew, wrong headline metric.
- Goal: surface each issue and apply all three fixes.

## Failure Mode Table

| Task | Observable Symptom | Hidden Cause(s) | Tempting Wrong Path |
|------|---------------------|-----------------|---------------------|
| Task 1 | Train accuracy is high, validation is flat | Validation transform missing normalization | Assume learning-rate or optimizer instability |
| Task 2 | Loss decreases while accuracy also decreases | Softmax is applied before `CrossEntropyLoss` | Tweak label smoothing or batch size instead |
| Task 3 | Train/val look strong, test collapses | Split leakage, class imbalance, wrong primary metric | Train longer, lower LR, or trust accuracy alone |

## What The Grader Rewards

The grader is deterministic and combines four ideas:

- evidence quality: did the agent reveal the task-critical clues?
- diagnosis quality: did it confirm the correct root causes?
- fix quality: did it apply the required fixes and avoid wrong ones?
- efficiency: did it finish without wasting the full step budget?

Blind guessing is intentionally weakened:

- wrong fixes are penalized
- premature fixes are penalized
- final completion requires the required fixes, required causes, and required evidence

## Actions

- `read_log(target)`
- `check_metric(target)`
- `inspect_config(target)`
- `propose_fix(target)`
- `apply_fix()`

## Observations

Fields include `task_id`, `description`, `difficulty`, `visible_logs`, `visible_metrics`, `visible_config`, `step_count`, `pending_fix`, `applied_fixes`, `message`, `done`.

## Reward Design

- revealing new useful information gives small positive reward
- unlocking task-critical evidence gives higher reward
- confirming a root cause gives a stronger reward bump
- proposing a correct fix after enough evidence gives partial credit
- applying the final required fix path finishes the episode
- repeated actions, wrong fixes, and premature fixes reduce overall score

## Local run

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 7860
```

## Testing & CI

```bash
pip install -r requirements.txt
pip install pytest pytest-cov ruff
ruff check .
pytest --cov=env --cov=server --cov=main --cov-report=term-missing
```

Every push and pull request to `main` runs lint (`ruff`) and the full test suite via GitHub Actions (see `.github/workflows/ci.yml`). Current coverage is ~83%, with `env/environment.py` - the core state machine - at 90%.

## Baseline (`inference.py`)

Uses the OpenAI-compatible client with env vars below. `HF_TOKEN` (or `OPENAI_API_KEY`) is required for inference runs.

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="your_token"
python inference.py
```

Required for the competition: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`.

Stdout format (per task): one `[START]`, one `[STEP]` per `step()`, one `[END]` after `close()`.

## Reference Baselines

The repo exposes two useful reference behaviors:

| Baseline | Purpose | Notes |
|----------|---------|-------|
| Scripted planner | Reproducible upper-reference trajectory | Used when fallback is enabled or through `/baseline` |
| Model-driven inference | Competition-facing baseline path | Uses the OpenAI-compatible client with `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` |

## Docker

```bash
docker build -t ml-debug-env .
docker run -p 7860:7860 \
  -e API_BASE_URL="https://router.huggingface.co/v1" \
  -e MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct" \
  -e HF_TOKEN="your_token" \
  ml-debug-env
```

## HTTP API

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Health |
| POST | `/reset` | New episode |
| POST | `/step` | One action |
| GET | `/state` | Current state |
| GET | `/tasks` | Task list |
| GET | `/grader` | Score snapshot |
| GET | `/baseline` | Scripted baseline |

Session behavior:
- If `X-Session-Id` header or `openenv_session_id` cookie is present, that session is used (isolated clients).
- If neither is present, the API falls back to a shared `default` session for stateless evaluators.

## Notes

- Typed models: `Action`, `Observation`, `Reward` (see `env/environment.py`).
- `openenv.yaml` describes tasks and endpoints.
- Graders are deterministic; scores are in `[0.0, 1.0]`.
- `tests/test_api.py` and `tests/test_graders.py` cover endpoint behavior and grading invariants.

## Baseline scores (scripted policy)

Current reproducible run (`python inference.py`):
- `task_1`: `0.98`
- `task_2`: `0.98`
- `task_3`: `0.96`

## Limitations

- The environment is intentionally deterministic and compact rather than fully open-ended.
- The current benchmark focuses on three failure families, not the full space of ML debugging issues.
- Session state is held in-process (an LRU-capped in-memory dict); it does not survive a restart and would need an external store (e.g. Redis) to run behind multiple replicas.
- Future extensions could add optimizer bugs, label corruption, data loader failures, and distributed training diagnostics.

## Acknowledgements

Built for the **Meta × PyTorch OpenEnv Hackathon**, in partnership with **Scaler School of Technology**.

## License

[MIT](LICENSE)
