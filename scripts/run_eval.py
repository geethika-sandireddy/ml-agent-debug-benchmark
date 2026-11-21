"""Evaluate agents on the real workloads and keep the results table up to date.

Reference agents (floor and ceiling, no API needed):
    python scripts/run_eval.py --agents noop,oracle --seeds 3 --update-readme

Language models (any OpenAI-compatible endpoint; set OPENAI_API_KEY and optionally OPENAI_BASE_URL):
    python scripts/run_eval.py --agents llm --models modelA,modelB,modelC --seeds 3 --update-readme

Every run writes results/eval_<agent>.json. The table is always rebuilt from ALL saved result
files, so earlier runs are kept and new models are added as rows. With --update-readme the table
is written between the EVAL_TABLE markers in README.md.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines import llm_agent, scripted
from env.real_env import RealMLDebugEnv
from workloads import list_workloads

START, END = "<!-- EVAL_TABLE_START -->", "<!-- EVAL_TABLE_END -->"
ORDER = {"noop": 0, "oracle": 1}


def build_table(results: list[dict], tasks: list[str]) -> str:
    """results: [{"agent": name, "seeds": n, "rewards": {task: [per-seed reward]}}]."""
    rows = sorted(results, key=lambda r: (ORDER.get(r["agent"], 2), r["agent"]))
    lines = ["| agent | seeds | " + " | ".join(tasks) + " | mean |", "|---|---|" + "---|" * (len(tasks) + 1)]
    for r in rows:
        means = [statistics.mean(r["rewards"][t]) for t in tasks]
        cells = " | ".join(f"{m:.2f}" for m in means)
        lines.append(f"| {r['agent']} | {r['seeds']} | {cells} | {statistics.mean(means):.2f} |")
    return "\n".join(lines)


def update_readme(readme: Path, table: str) -> None:
    text = readme.read_text()
    if START not in text or END not in text:
        raise SystemExit(f"{readme} has no {START} / {END} markers")
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    readme.write_text(f"{head}{START}\n{table}\n{END}{tail}")


def load_results(out: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(out.glob("eval_*.json"))]


def run_agent(label: str, fn, tasks: list[str], seeds: int, out: Path) -> None:
    env = RealMLDebugEnv()
    rewards = {t: [fn(env, t, s) for s in range(seeds)] for t in tasks}
    env.close()
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
    (out / f"eval_{safe}.json").write_text(json.dumps({"agent": label, "seeds": seeds, "rewards": rewards}, indent=2))
    print(f"{label}: mean reward {statistics.mean(statistics.mean(v) for v in rewards.values()):.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--agents", default="noop,oracle", help="comma list of: noop, oracle, llm")
    ap.add_argument("--models", default=os.environ.get("MODEL_NAME", ""), help="comma list of model names for --agents llm")
    ap.add_argument("--table-only", action="store_true", help="rebuild the table from saved results, run nothing")
    ap.add_argument("--update-readme", action="store_true")
    args = ap.parse_args()
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    tasks = list_workloads()
    if not args.table_only:
        for name in args.agents.split(","):
            if name == "noop":
                run_agent("noop", scripted.run_noop, tasks, args.seeds, out)
            elif name == "oracle":
                run_agent("oracle", scripted.run_oracle, tasks, args.seeds, out)
            elif name == "llm":
                if not llm_agent.available():
                    raise SystemExit("Set OPENAI_API_KEY (and OPENAI_BASE_URL for non-OpenAI providers) to evaluate models.")
                models = [m for m in args.models.split(",") if m] or ["gpt-4o-mini"]
                for model in models:
                    os.environ["MODEL_NAME"] = model
                    run_agent(f"llm:{model}", llm_agent.run_llm, tasks, args.seeds, out)
    table = build_table(load_results(out), tasks)
    (out / "eval_table.md").write_text(table + "\n")
    print(table)
    if args.update_readme:
        update_readme(ROOT / "README.md", table)


if __name__ == "__main__":
    main()
