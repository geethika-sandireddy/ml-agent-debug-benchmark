"""Environment backed by real PyTorch workloads (v2). The v1 simulator (env/environment.py)
remains available as the legacy mode."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from grader import grade_workload
from sandbox.runner import RunResult, run_script
from workloads import diagnosis_labels, list_workloads, load_config, prepare

MAX_STEPS = 15
MAX_RUNS = 3
EDITABLE = {"train.py", "common.py"}  # editing common.py is allowed but scores 0 at grading


class RealAction(BaseModel):
    action_type: Literal["read_file", "edit_file", "run_training", "inspect_metrics", "submit_diagnosis"]
    path: str = ""
    old: str = ""
    new: str = ""
    label: str = ""


class RealObservation(BaseModel):
    task_id: str
    description: str
    files: list[str] = Field(default_factory=list)
    log_tail: str = ""
    metric_summary: str = ""
    content: str = ""
    diagnosis_options: list[str] = Field(default_factory=list)
    step_count: int = 0
    runs_used: int = 0
    message: str = ""
    done: bool = False


class RealStepResult(BaseModel):
    observation: RealObservation
    reward: float
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)


def summarize_metrics(rows: list[dict], keep: int = 4) -> str:
    if not rows:
        return "(no metrics)"
    shown = rows if len(rows) <= keep * 2 else rows[:keep] + [{"...": "..."}] + rows[-keep:]
    return "\n".join(str(r) for r in shown)


class RealMLDebugEnv:
    def __init__(self) -> None:
        self.task_id: str | None = None
        self.seed = 0
        self.workdir: Path | None = None
        self._tmp: Path | None = None
        self.steps = self.edits = self.runs = 0
        self.last: RunResult | None = None
        self.done = False
        self.final_reward = 0.0

    # --- lifecycle ---
    def reset(self, task_id: str | None = None, seed: int = 0) -> RealObservation:
        self.close()
        self.task_id = task_id or list_workloads()[0]
        cfg = load_config(self.task_id)
        self.seed = seed
        self._tmp = Path(tempfile.mkdtemp(prefix="mlenv_"))
        self.workdir = prepare(self.task_id, "buggy", self._tmp / "w")
        self.steps = self.edits = self.runs = 0
        self.done, self.final_reward = False, 0.0
        self.last = self._run()  # initial run does not count against the agent's budget
        return self._obs(f"Episode started. Symptom: {cfg['symptom']}.")

    def close(self) -> None:
        if self._tmp:
            shutil.rmtree(self._tmp, ignore_errors=True)
        self._tmp = self.workdir = None

    def _run(self) -> RunResult:
        return run_script(self.workdir, extra_env={"SEED": str(self.seed)})

    def _obs(self, message: str = "", content: str = "") -> RealObservation:
        files = sorted(p.name for p in self.workdir.iterdir() if p.is_file() and p.suffix == ".py") if self.workdir else []
        last = self.last
        return RealObservation(
            task_id=self.task_id or "",
            description=load_config(self.task_id)["symptom"] if self.task_id else "",
            files=files,
            log_tail=(last.stdout[-1200:] + last.stderr[-600:]) if last else "",
            metric_summary=summarize_metrics(last.metrics) if last else "",
            content=content,
            diagnosis_options=diagnosis_labels(),
            step_count=self.steps,
            runs_used=self.runs,
            message=message,
            done=self.done,
        )

    def state(self) -> dict[str, Any]:
        return {"task_id": self.task_id, "steps_used": self.steps, "edits": self.edits,
                "runs": self.runs, "max_steps": MAX_STEPS, "max_runs": MAX_RUNS, "done": self.done}

    # --- stepping ---
    def _safe_path(self, name: str) -> Path | None:
        if "/" in name or "\\" in name or name in ("", ".", ".."):
            return None
        p = self.workdir / name
        return p if p.is_file() and p.suffix == ".py" else None

    def step(self, action: RealAction) -> RealStepResult:
        if not self.workdir:
            raise RuntimeError("Environment has not been reset.")
        if self.done:
            return RealStepResult(observation=self._obs("Episode already finished."), reward=0.0, done=True,
                                  info={"error": "episode_already_done"})
        self.steps += 1
        msg, content, err, rew = "", "", None, 0.0
        t = action.action_type
        if t == "read_file":
            p = self._safe_path(action.path)
            if p is None:
                err, msg = "unknown_file", f"'{action.path}' is not a readable file."
            else:
                content, msg = p.read_text(), f"Read {action.path}."
        elif t == "edit_file":
            p = self._safe_path(action.path)
            text = p.read_text() if p else ""
            if p is None or action.path not in EDITABLE:
                err, msg = "unknown_file", f"'{action.path}' cannot be edited."
            elif not action.old or text.count(action.old) != 1:
                err, msg = "bad_edit", "old text must match exactly once."
            else:
                p.write_text(text.replace(action.old, action.new))
                self.edits += 1
                msg = f"Edited {action.path}."
        elif t == "run_training":
            if self.runs >= MAX_RUNS:
                err, msg = "run_budget_exhausted", f"Run budget of {MAX_RUNS} used."
            else:
                self.runs += 1
                self.last = self._run()
                msg = "Training run finished." if self.last.ok else "Training run failed or timed out."
        elif t == "inspect_metrics":
            content, msg = summarize_metrics(self.last.metrics, keep=50) if self.last else "(no run)", "Metrics from last run."
        elif t == "submit_diagnosis":
            res = grade_workload(self.task_id, self.workdir, action.label, self.steps, MAX_STEPS, self.seed)
            self.done, self.final_reward = True, res.reward
            rew = res.reward
            msg = "Submitted."
            return RealStepResult(observation=self._obs(msg), reward=rew, done=True,
                                  info={"diagnosis_correct": res.diagnosis_correct, "fix_verified": res.fix_verified,
                                        "violation": res.violation})
        if self.steps >= MAX_STEPS:
            self.done = True
            msg += " Step budget exhausted without a submission."
        return RealStepResult(observation=self._obs(msg, content), reward=rew, done=self.done,
                              info={"error": err})
