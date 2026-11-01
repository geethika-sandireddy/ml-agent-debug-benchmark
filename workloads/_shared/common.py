"""Shared helpers for every workload. PROTECTED: the grader hashes this file.

Data: scikit-learn digits (bundled, no download). A hidden test pool is carved out with a
fixed permutation and is never returned to training scripts; only `finalize` touches it,
and only when the grader sets HIDDEN_EVAL.
"""

import json
import os
import random

import numpy as np
import torch
from sklearn.datasets import load_digits

HIDDEN_SEED = 9999
HIDDEN_SIZE = 400


def get_seed(default: int = 0) -> int:
    """Seed for this run (the environment/tests set SEED)."""
    return int(os.environ.get("SEED", default))


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def _split_indices(n: int):
    perm = np.random.RandomState(HIDDEN_SEED).permutation(n)
    return perm[HIDDEN_SIZE:], perm[:HIDDEN_SIZE]  # pool, hidden


def load_pool(seed: int):
    """Return (X, y, order): the non-hidden pool as float32/int64 tensors and a seeded
    shuffled index order the script can slice into train/val."""
    d = load_digits()
    pool_idx, _ = _split_indices(len(d.target))
    X = torch.tensor(d.data[pool_idx], dtype=torch.float32)
    y = torch.tensor(d.target[pool_idx], dtype=torch.long)
    order = np.random.RandomState(seed).permutation(len(pool_idx))
    return X, y, order


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(1) == y).float().mean().item()


def log_metrics(**kw) -> None:
    kw = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in kw.items()}
    print(json.dumps(kw))
    path = os.environ.get("METRICS_PATH")
    if path:
        with open(path, "a") as f:
            f.write(json.dumps(kw) + "\n")


def finalize(predict_fn, train_idx=None, val_idx=None) -> None:
    """Called at the end of every training script. `predict_fn(raw_X) -> logits` must apply
    the script's own preprocessing and mode. Writes hidden-test accuracy only when the
    grader asks for it; also reports train/val overlap when indices are given."""
    out = os.environ.get("HIDDEN_EVAL_OUT")
    if not out:
        return
    d = load_digits()
    _, hid = _split_indices(len(d.target))
    X = torch.tensor(d.data[hid], dtype=torch.float32)
    y = torch.tensor(d.target[hid], dtype=torch.long)
    with torch.no_grad():
        acc = accuracy(predict_fn(X), y)
    res = {"hidden_acc": acc}
    if train_idx is not None and val_idx is not None:
        res["overlap"] = len(set(map(int, train_idx)) & set(map(int, val_idx)))
    with open(out, "w") as f:
        json.dump(res, f)
