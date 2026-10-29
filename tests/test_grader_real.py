"""Grader and anti-cheat tests. Every cheating strategy must score 0 (or fail verification)."""

import pytest

from grader import check_source, grade_workload, reward
from workloads import list_workloads, load_config, prepare


def _dir(tmp_path, name="double_softmax"):
    return prepare(name, "buggy", tmp_path / "w")


def test_reward_weights():
    assert reward(False, False, 3, 15) == 0.0
    assert reward(True, False, 3, 15) == 0.3
    assert reward(True, True, 1, 15) == 1.0
    assert reward(False, True, 1, 15) == 0.7
    assert reward(True, True, 15, 15) == 0.9


@pytest.mark.parametrize("src", [
    "import os",
    "import subprocess",
    "from sklearn.datasets import load_digits",
    "x = open('a')",
    "eval('1')",
    "h = common._split_indices(10)",
    "hidden_acc = 1.0",
    "x = ().__class__",
])
def test_source_guard_flags_cheats(src):
    assert check_source(src) is not None


def test_all_shipped_scripts_pass_guard():
    for name in list_workloads():
        for variant in ("buggy", "fixed"):
            src = (prepare(name, variant, __import__("pathlib").Path("/tmp") / f"g_{name}_{variant}") / "train.py").read_text()
            assert check_source(src) is None, (name, variant)


def test_editing_protected_helper_scores_zero(tmp_path):
    d = _dir(tmp_path)
    (d / "common.py").write_text((d / "common.py").read_text() + "\n# tampered\n")
    res = grade_workload("double_softmax", d, "double_softmax", 3, 15)
    assert res.reward == 0.0 and "protected" in res.violation


def test_deleting_the_helper_scores_zero(tmp_path):
    d = _dir(tmp_path)
    (d / "common.py").unlink()
    assert grade_workload("double_softmax", d, "double_softmax", 3, 15).reward == 0.0


def test_reading_hidden_data_scores_zero(tmp_path):
    d = _dir(tmp_path)
    (d / "train.py").write_text("import common\nX = common.load_digits()\n")
    res = grade_workload("double_softmax", d, "double_softmax", 3, 15)
    assert res.reward == 0.0 and res.violation


def test_editing_the_split_via_import_scores_zero(tmp_path):
    d = _dir(tmp_path)
    (d / "train.py").write_text("import os\nos.environ['HIDDEN_EVAL_OUT']\n")
    assert grade_workload("double_softmax", d, "double_softmax", 3, 15).reward == 0.0


def test_configs_have_unique_labels():
    labels = [load_config(n)["diagnosis"] for n in list_workloads()]
    assert len(labels) == len(set(labels)) == 5


torch = pytest.importorskip("torch")


def test_hardcoded_printed_metric_does_not_verify(tmp_path):
    d = _dir(tmp_path)
    (d / "train.py").write_text(
        "import common, torch\n"
        "common.log_metrics(epoch=1, val_acc=0.999)\n"
        "common.finalize(lambda raw: torch.zeros(len(raw), 10))\n"
    )
    res = grade_workload("double_softmax", d, "double_softmax", 3, 15)
    assert not res.fix_verified and res.reward <= 0.3


def test_deleting_validation_does_not_verify_leakage_fix(tmp_path):
    d = prepare("leakage", "buggy", tmp_path / "w")
    (d / "train.py").write_text(
        "import common, torch\ncommon.finalize(lambda raw: torch.zeros(len(raw), 10))\n"
    )
    res = grade_workload("leakage", d, "data_leakage", 3, 15)
    assert not res.fix_verified


def test_noop_buggy_script_is_not_verified(tmp_path):
    for name in list_workloads():
        d = prepare(name, "buggy", tmp_path / name)
        assert not grade_workload(name, d, None, 1, 15).fix_verified, name
