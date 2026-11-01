"""Deterministic grader for real workloads.

reward = 0.3 * correct diagnosis + 0.6 * verified fix + 0.1 * efficiency (only if fixed).
A verified fix = the agent's final train.py, rerun in a CLEAN copy with a pristine helper,
reaches the hidden-test threshold (and has zero train/val overlap where required).
Tampering with protected files, or source that touches forbidden names, scores 0.
This is a benchmark guardrail, not a security boundary.
"""

from __future__ import annotations

import ast
import hashlib
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from sandbox.runner import run_script
from workloads import PROTECTED_FILES, load_config, prepare, protected_hashes

FORBIDDEN_IMPORTS = {
    "os", "sys", "subprocess", "importlib", "builtins", "ctypes", "shutil", "pathlib",
    "glob", "sklearn", "socket", "pickle", "inspect", "gc",
}
FORBIDDEN_CALLS = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "getattr", "setattr"}
FORBIDDEN_SUBSTRINGS = ("hidden", "HIDDEN", "_split_indices", "load_digits")


@dataclass
class GradeResult:
    reward: float = 0.0
    diagnosis_correct: bool = False
    fix_verified: bool = False
    violation: str | None = None
    details: dict = field(default_factory=dict)


def check_source(source: str) -> str | None:
    """Return a violation message, or None if the script looks clean."""
    for token in FORBIDDEN_SUBSTRINGS:
        if token in source:
            return f"forbidden name '{token}'"
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return f"syntax error: {exc.msg}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in FORBIDDEN_IMPORTS:
                    return f"forbidden import '{a.name}'"
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in FORBIDDEN_IMPORTS:
                return f"forbidden import '{node.module}'"
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS:
                return f"forbidden call '{node.func.id}'"
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return f"forbidden dunder access '{node.attr}'"
    return None


def reward(diagnosis_correct: bool, fix_verified: bool, steps_used: int, max_steps: int) -> float:
    score = 0.3 * diagnosis_correct + 0.6 * fix_verified
    if fix_verified:
        score += 0.1 * max(0.0, 1.0 - (steps_used - 1) / max(1, max_steps - 1))
    return round(score, 3)


def verify_fix(name: str, workdir: Path, seed: int = 0, timeout_s: int = 40) -> tuple[bool, dict]:
    cfg = load_config(name)
    src = (workdir / "train.py").read_text()
    bad = check_source(src)
    if bad:
        return False, {"violation": bad}
    for f in PROTECTED_FILES:
        p = workdir / f
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != protected_hashes()[f]:
            return False, {"violation": f"protected file '{f}' was modified"}
    with tempfile.TemporaryDirectory(prefix="mlgrade_") as tmp:
        clean = Path(tmp) / "w"
        prepare(name, "buggy", clean)  # pristine helper files
        (clean / "train.py").write_text(src)
        out = clean / "hidden.json"
        res = run_script(clean, timeout_s=timeout_s,
                         extra_env={"SEED": str(seed), "HIDDEN_EVAL_OUT": str(out)})
        if not res.ok or not out.exists():
            return False, {"error": "clean rerun failed or finalize() was not called",
                           "stderr": res.stderr[-500:]}
        hidden = json.loads(out.read_text())
    ok = cfg["threshold"] is not None and hidden["hidden_acc"] >= cfg["threshold"]
    if cfg.get("require_zero_overlap"):
        ok = ok and hidden.get("overlap") == 0
    return ok, {"hidden": hidden, "threshold": cfg["threshold"]}


def grade_workload(name: str, workdir: Path, diagnosis: str | None, steps_used: int,
                   max_steps: int, seed: int = 0) -> GradeResult:
    cfg = load_config(name)
    ok, details = verify_fix(name, workdir, seed)
    if "violation" in details:
        return GradeResult(0.0, False, False, details["violation"], details)
    correct = diagnosis == cfg["diagnosis"]
    return GradeResult(reward(correct, ok, steps_used, max_steps), correct, ok, None, details)
