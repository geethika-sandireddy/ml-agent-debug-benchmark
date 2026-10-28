"""LLM agent over an OpenAI-compatible API. Needs OPENAI_API_KEY (and optionally
OPENAI_BASE_URL, MODEL_NAME). Without a key this run is simply not performed."""

from __future__ import annotations

import json
import os

from env.real_env import MAX_STEPS, RealAction, RealMLDebugEnv

SYSTEM = (
    "You debug a failing PyTorch training script. Reply with ONE JSON object per turn, no prose: "
    '{"action_type": "read_file|edit_file|run_training|inspect_metrics|submit_diagnosis", '
    '"path": "train.py", "old": "exact text to replace (must match once)", "new": "replacement", '
    '"label": "one of diagnosis_options"}. Read the code, fix the bug with edit_file, optionally '
    "run_training to confirm, then submit_diagnosis with the label that names the bug."
)


def available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def run_llm(env: RealMLDebugEnv, task_id: str, seed: int = 0) -> float:
    from openai import OpenAI

    client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL"))
    model = os.environ.get("MODEL_NAME", "gpt-4o-mini")
    obs = env.reset(task_id, seed)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": obs.model_dump_json()}]
    for _ in range(MAX_STEPS):
        reply = client.chat.completions.create(model=model, messages=messages, temperature=0).choices[0].message.content or ""
        messages.append({"role": "assistant", "content": reply})
        try:
            action = RealAction(**json.loads(reply[reply.index("{"): reply.rindex("}") + 1]))
        except (ValueError, TypeError):
            messages.append({"role": "user", "content": "Invalid JSON action. Reply with one JSON object."})
            continue
        result = env.step(action)
        if result.done:
            return result.reward
        messages.append({"role": "user", "content": result.observation.model_dump_json()})
    return 0.0
