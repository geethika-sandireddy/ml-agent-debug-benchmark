"""Floor and ceiling agents for the real environment (no LLM needed)."""

from __future__ import annotations

from env.real_env import RealAction, RealMLDebugEnv
from workloads import load_config

# Known fixes as exact edits to train.py (the oracle's "knowledge").
ORACLE_EDITS: dict[str, list[tuple[str, str]]] = {
    "double_softmax": [(", nn.Softmax(dim=1))", ")")],
    "norm_mismatch": [
        ("Xva, yva = X[va], y[va]", "Xva, yva = (X[va] - mean) / std, y[va]"),
        ("common.finalize(lambda raw: model(raw))", "common.finalize(lambda raw: model((raw - mean) / std))"),
    ],
    "leakage": [("order[:100]]", "order[250:350]]")],
    "scheduler_bug": [
        ("        sched.step()\n", ""),
        ("    model.eval()\n", "    sched.step()\n    model.eval()\n"),
    ],
    "missing_eval": [("    with torch.no_grad():\n        val_acc", "    model.eval()\n    with torch.no_grad():\n        val_acc")],
}


def run_noop(env: RealMLDebugEnv, task_id: str, seed: int = 0) -> float:
    """Floor: look at the task, change nothing, submit no diagnosis."""
    env.reset(task_id, seed)
    return env.step(RealAction(action_type="submit_diagnosis", label="")).reward


def run_oracle(env: RealMLDebugEnv, task_id: str, seed: int = 0) -> float:
    """Ceiling: read the script, apply the known fix, rerun once, submit the right label."""
    env.reset(task_id, seed)
    env.step(RealAction(action_type="read_file", path="train.py"))
    for old, new in ORACLE_EDITS[task_id]:
        env.step(RealAction(action_type="edit_file", path="train.py", old=old, new=new))
    env.step(RealAction(action_type="run_training"))
    return env.step(RealAction(action_type="submit_diagnosis", label=load_config(task_id)["diagnosis"])).reward
