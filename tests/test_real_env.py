import pytest

from baselines.scripted import run_noop, run_oracle
from env.real_env import MAX_RUNS, MAX_STEPS, RealAction, RealMLDebugEnv
from workloads import list_workloads, load_config

pytest.importorskip("torch")


@pytest.fixture
def env():
    e = RealMLDebugEnv()
    yield e
    e.close()


def test_reset_shows_real_output_without_leaking_answers(env):
    obs = env.reset("double_softmax", seed=0)
    assert "train.py" in obs.files and "config.json" not in obs.files
    assert obs.metric_summary != "(no metrics)"
    cfg = load_config("double_softmax")
    visible = obs.model_dump_json(exclude={"diagnosis_options"})
    assert cfg["diagnosis"] not in visible and obs.task_id == "task_1"
    assert "BUG" not in env.step(RealAction(action_type="read_file", path="train.py")).observation.content


def test_read_rejects_paths_outside_workdir(env):
    env.reset("double_softmax")
    for bad in ("../x.py", "/etc/passwd", "config.json", ""):
        r = env.step(RealAction(action_type="read_file", path=bad))
        assert r.info["error"] == "unknown_file"


def test_edit_must_match_exactly_once(env):
    env.reset("double_softmax")
    r = env.step(RealAction(action_type="edit_file", path="train.py", old="nonexistent text", new="x"))
    assert r.info["error"] == "bad_edit" and env.state()["edits"] == 0


def test_run_budget_is_enforced(env):
    env.reset("double_softmax")
    errs = [env.step(RealAction(action_type="run_training")).info["error"] for _ in range(MAX_RUNS + 1)]
    assert errs[:MAX_RUNS] == [None] * MAX_RUNS and errs[-1] == "run_budget_exhausted"


def test_step_cap_ends_episode(env):
    env.reset("double_softmax")
    for _ in range(MAX_STEPS):
        r = env.step(RealAction(action_type="inspect_metrics"))
    assert r.done and env.state()["done"]


def test_state_counts(env):
    env.reset("double_softmax")
    env.step(RealAction(action_type="run_training"))
    s = env.state()
    assert s["steps_used"] == 1 and s["runs"] == 1


@pytest.mark.parametrize("task", list_workloads())
def test_oracle_is_near_ceiling_and_noop_is_floor(env, task):
    assert run_oracle(env, task, 0) >= 0.9
    assert run_noop(env, task, 0) == 0.0
