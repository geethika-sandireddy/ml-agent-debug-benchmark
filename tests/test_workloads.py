"""Each bug must make the buggy run measurably worse than the fixed run across 5 seeds,
and each run must finish in under 20 seconds."""

import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("torch")

from sandbox.runner import run_script
from workloads import list_workloads, load_config, prepare

SEEDS = range(5)
MAX_SECONDS = 20


def _run(name, variant, seed, tmp_path):
    d = prepare(name, variant, tmp_path / f"{name}_{variant}_{seed}")
    out = d / "hidden.json"
    t0 = time.time()
    res = run_script(d, extra_env={"SEED": str(seed), "HIDDEN_EVAL_OUT": str(out)})
    return res, json.loads(out.read_text()), time.time() - t0


@pytest.fixture(scope="module", params=list_workloads())
def runs(request, tmp_path_factory):
    name = request.param
    tmp = tmp_path_factory.mktemp(name)
    jobs = [(v, s) for v in ("buggy", "fixed") for s in SEEDS]
    with ThreadPoolExecutor(max_workers=1) as ex:  # serial so timings are fair
        results = list(ex.map(lambda j: _run(name, j[0], j[1], tmp), jobs))
    return name, {j: r for j, r in zip(jobs, results, strict=True)}


def test_runs_succeed_and_are_fast(runs):
    _, res = runs
    for r, _hidden, secs in res.values():
        assert r.ok, r.stderr
        assert secs < MAX_SECONDS


def test_runs_are_deterministic_per_seed(runs, tmp_path):
    name, res = runs
    _r2, hidden2, _ = _run(name, "buggy", 0, tmp_path)
    assert hidden2["hidden_acc"] == res[("buggy", 0)][1]["hidden_acc"]


def test_buggy_is_worse_than_fixed_across_5_seeds(runs):
    name, res = runs
    cfg = load_config(name)
    buggy = [res[("buggy", s)][1] for s in SEEDS]
    fixed = [res[("fixed", s)][1] for s in SEEDS]
    if name == "leakage":
        # Same training set, so hidden accuracy matches; the leak shows up as inflated
        # validation accuracy and non-zero train/val overlap.
        gap = lambda rs: [r[0].metrics[-1]["val_acc"] for r in rs]
        b = gap([res[("buggy", s)] for s in SEEDS])
        f = gap([res[("fixed", s)] for s in SEEDS])
        assert statistics.mean(b) - statistics.mean(f) > 0.03
        assert all(x["overlap"] > 0 for x in buggy) and all(x["overlap"] == 0 for x in fixed)
        assert min(x["hidden_acc"] for x in fixed) >= cfg["threshold"]
    else:
        assert statistics.mean(x["hidden_acc"] for x in fixed) - statistics.mean(x["hidden_acc"] for x in buggy) > 0.03
        assert max(x["hidden_acc"] for x in buggy) < cfg["threshold"] <= min(x["hidden_acc"] for x in fixed)
