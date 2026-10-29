"""Run baselines over tasks and seeds; write results/eval_<agent>.json and results/eval_table.md.
Only real runs are written. The LLM agent is skipped (and reported as not run) without a key."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines import llm_agent, scripted
from env.real_env import RealMLDebugEnv
from workloads import list_workloads

AGENTS = {"noop": scripted.run_noop, "oracle": scripted.run_oracle, "llm": llm_agent.run_llm}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--agents", default="noop,oracle,llm")
    args = ap.parse_args()
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    tasks = list_workloads()
    table = {}
    for name in args.agents.split(","):
        if name == "llm" and not llm_agent.available():
            table[name] = None
            print("llm: not performed (no OPENAI_API_KEY)")
            continue
        env = RealMLDebugEnv()
        rows = {t: [AGENTS[name](env, t, s) for s in range(args.seeds)] for t in tasks}
        env.close()
        (out / f"eval_{name}.json").write_text(json.dumps({"agent": name, "seeds": args.seeds, "rewards": rows}, indent=2))
        table[name] = {t: statistics.mean(v) for t, v in rows.items()}
    lines = ["| agent | " + " | ".join(tasks) + " | mean |", "|---|" + "---|" * (len(tasks) + 1)]
    for name, res in table.items():
        if res is None:
            lines.append(f"| {name} | " + " | ".join("not run" for _ in tasks) + " | not run |")
        else:
            lines.append(f"| {name} | " + " | ".join(f"{res[t]:.2f}" for t in tasks) + f" | {statistics.mean(res.values()):.2f} |")
    (out / "eval_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
