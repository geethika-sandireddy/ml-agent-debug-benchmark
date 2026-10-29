"""Workload registry: one folder per injected bug (buggy.py, fixed.py, config.json)."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).parent
SHARED = ROOT / "_shared"
PROTECTED_FILES = ("common.py",)


def list_workloads() -> list[str]:
    return sorted(p.name for p in ROOT.iterdir() if (p / "config.json").exists())


def public_id(name: str) -> str:
    """Neutral id shown to agents (task_1..task_N); the folder name can reveal the bug."""
    return f"task_{list_workloads().index(name) + 1}"


def resolve(task: str) -> str:
    """Accept either a public id (task_3) or an internal folder name."""
    for n in list_workloads():
        if task in (n, public_id(n)):
            return n
    raise KeyError(f"Unknown workload '{task}'")


def load_config(name: str) -> dict:
    path = ROOT / name / "config.json"
    if not path.exists():
        raise KeyError(f"Unknown workload '{name}'")
    return json.loads(path.read_text())


def diagnosis_labels() -> list[str]:
    return sorted(load_config(n)["diagnosis"] for n in list_workloads())


def prepare(name: str, variant: str, dest: Path) -> Path:
    """Create a run directory: shared helper files plus the chosen variant as train.py.
    config.json is NOT copied, so the agent never sees thresholds or the diagnosis label."""
    if variant not in ("buggy", "fixed"):
        raise ValueError("variant must be 'buggy' or 'fixed'")
    load_config(name)
    dest.mkdir(parents=True, exist_ok=True)
    for f in PROTECTED_FILES:
        shutil.copy(SHARED / f, dest / f)
    shutil.copy(ROOT / name / f"{variant}.py", dest / "train.py")
    return dest


def protected_hashes() -> dict[str, str]:
    return {f: hashlib.sha256((SHARED / f).read_bytes()).hexdigest() for f in PROTECTED_FILES}
