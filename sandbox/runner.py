"""Run a workload script in a throwaway directory with a timeout and resource limits.

NOT a security boundary: it limits runaway scripts, it does not stop a malicious one.
CPU/memory limits use `resource` and only apply on Linux/macOS; on Windows only the
timeout applies.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

try:  # POSIX only
    import resource
except ImportError:  # pragma: no cover
    resource = None  # type: ignore[assignment]

DEFAULT_TIMEOUT_S = 30
DEFAULT_MEM_BYTES = 2 * 1024**3
DEFAULT_CPU_S = 60
OUTPUT_LIMIT = 20_000
METRICS_FILE = "metrics.jsonl"


@dataclass
class RunResult:
    returncode: int | None
    stdout: str
    stderr: str
    metrics: list[dict] = field(default_factory=list)
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def _limits(mem_bytes: int, cpu_s: int):
    def apply() -> None:
        if resource is None:
            return
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))

    return apply


def _read_metrics(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def run_script(
    workdir: Path,
    script: str = "train.py",
    timeout_s: int = DEFAULT_TIMEOUT_S,
    mem_bytes: int = DEFAULT_MEM_BYTES,
    cpu_s: int = DEFAULT_CPU_S,
    extra_env: dict[str, str] | None = None,
) -> RunResult:
    """Run `script` inside `workdir` (already a private copy). Does not delete workdir."""
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "METRICS_PATH": str(workdir / METRICS_FILE),
    }
    env.update(extra_env or {})
    preexec = _limits(mem_bytes, cpu_s) if resource is not None else None
    try:
        proc = subprocess.run(
            [sys.executable, script],
            cwd=workdir,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            preexec_fn=preexec,
        )
        return RunResult(
            proc.returncode,
            proc.stdout[-OUTPUT_LIMIT:],
            proc.stderr[-OUTPUT_LIMIT:],
            _read_metrics(workdir / METRICS_FILE),
        )
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or b"")
        err = (exc.stderr or b"")
        out = out.decode() if isinstance(out, bytes) else out
        err = err.decode() if isinstance(err, bytes) else err
        return RunResult(None, out[-OUTPUT_LIMIT:], err[-OUTPUT_LIMIT:],
                         _read_metrics(workdir / METRICS_FILE), timed_out=True)


def run_in_temp_copy(src: Path, script: str = "train.py", **kwargs) -> RunResult:
    """Copy `src` folder to a temp dir, run `script`, delete the temp dir."""
    tmp = Path(tempfile.mkdtemp(prefix="mlbench_"))
    try:
        dest = tmp / "w"
        shutil.copytree(src, dest)
        return run_script(dest, script, **kwargs)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
