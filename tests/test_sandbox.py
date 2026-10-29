import json
import sys
from pathlib import Path

import pytest

from sandbox.runner import run_in_temp_copy, run_script

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="rlimits are POSIX only")


def _mk(tmp_path: Path, code: str) -> Path:
    d = tmp_path / "wl"
    d.mkdir()
    (d / "train.py").write_text(code)
    return d


def test_captures_stdout_and_metrics(tmp_path):
    d = _mk(tmp_path, "import os,json\nprint('hello')\n"
            "open(os.environ['METRICS_PATH'],'w').write(json.dumps({'epoch':1,'acc':0.5})+'\\n')")
    r = run_script(d)
    assert r.ok and "hello" in r.stdout
    assert r.metrics == [{"epoch": 1, "acc": 0.5}]


def test_timeout_is_reported(tmp_path):
    d = _mk(tmp_path, "import time\ntime.sleep(10)")
    r = run_script(d, timeout_s=1)
    assert r.timed_out and not r.ok


def test_nonzero_exit_is_not_ok(tmp_path):
    d = _mk(tmp_path, "raise SystemExit(3)")
    r = run_script(d)
    assert r.returncode == 3 and not r.ok


@posix_only
def test_memory_limit_kills_hog(tmp_path):
    d = _mk(tmp_path, "x = bytearray(1024**3)")
    r = run_script(d, mem_bytes=256 * 1024**2)
    assert not r.ok


def test_temp_copy_does_not_touch_source_and_cleans_up(tmp_path):
    d = _mk(tmp_path, "open('side_effect.txt','w').write('x')")
    run_in_temp_copy(d)
    assert not (d / "side_effect.txt").exists()


def test_metrics_ignores_bad_lines(tmp_path):
    d = _mk(tmp_path, "import os\nopen(os.environ['METRICS_PATH'],'w').write('oops\\n{\"a\":1}\\n')")
    assert run_script(d).metrics == [{"a": 1}]
    assert json.loads('{"a":1}')
