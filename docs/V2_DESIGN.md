# v2 design: real PyTorch workloads

v1 (tag `v1-hackathon`) shows the agent scripted fake logs for 3 tasks. v2 makes each task a
real PyTorch training script with an injected bug. The environment runs it, shows the agent
the real output, lets it edit code and rerun, then grades whether the bug was really fixed.

## Components
- `workloads/<bug>/`: `buggy.py`, `fixed.py`, `config.json` (seed, pass threshold, diagnosis label).
  Shared helper: sklearn digits (bundled, no download), tiny MLP, fixed seeds, deterministic
  torch, one CPU thread, JSON-lines metrics. A hidden test split (different seed) is used only
  by the grader. Each run takes under 20 s.
- `sandbox/`: copies a workload to a temp dir, runs it in a subprocess with timeout and
  CPU/memory limits (POSIX only), captures output and metrics, deletes the dir.
  Not a security boundary.
- `env/` keeps the OpenEnv API. New actions: read_file, edit_file, run_training,
  inspect_metrics, submit_diagnosis. Episodes capped (15 steps, 3 runs). The v1 simulator stays
  as a legacy mode.
- `grader/`: deterministic reward = correct diagnosis + verified fix on hidden split + small
  efficiency bonus. Protected files are hashed at reset; editing them scores 0.
- `baselines/`: no-op (floor), scripted oracle (ceiling), LLM agent via OpenAI-compatible API.

## The five bugs
double softmax, normalization mismatch, validation leakage, scheduler stepping per batch,
missing model.eval() with dropout.

## Honesty rules
No invented numbers. If no API key is available, the LLM baseline is reported as "not run".

## Order of work
1. design + baseline test count (v1: 26 tests passing)
2. sandbox runner, 3. workload helper, 4. each bug with its test, 5. env integration,
6. grader + anti-cheat tests, 7. baselines/eval, 8. Docker/CI, 9. README, 10. release.
