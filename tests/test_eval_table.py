import json

from scripts.run_eval import END, START, build_table, load_results, update_readme

TASKS = ["a", "b"]


def res(agent, seeds, a, b):
    return {"agent": agent, "seeds": seeds, "rewards": {"a": a, "b": b}}


def test_table_orders_reference_agents_first_and_averages_seeds():
    rows = [res("llm:m1", 3, [1.0, 0.0, 0.5], [0.3, 0.3, 0.3]), res("oracle", 1, [1.0], [0.9]), res("noop", 1, [0.0], [0.0])]
    lines = build_table(rows, TASKS).splitlines()
    assert [ln.split("|")[1].strip() for ln in lines[2:]] == ["noop", "oracle", "llm:m1"]
    assert lines[4].startswith("| llm:m1 | 3 | 0.50 | 0.30 | 0.40 |")


def test_update_readme_replaces_only_the_marked_block(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(f"before\n{START}\nold table\n{END}\nafter\n")
    update_readme(readme, "NEW")
    assert readme.read_text() == f"before\n{START}\nNEW\n{END}\nafter\n"


def test_load_results_reads_every_saved_run(tmp_path):
    (tmp_path / "eval_x.json").write_text(json.dumps(res("x", 1, [1], [1])))
    (tmp_path / "eval_y.json").write_text(json.dumps(res("y", 1, [0], [0])))
    assert [r["agent"] for r in load_results(tmp_path)] == ["x", "y"]
