import numpy as np
import torch
import torch.nn as nn

import common

SEED = 0
common.set_determinism(SEED)
X, y, order = common.load_pool(SEED)
train_idx = order[:150]
val_idx = np.concatenate([order[150:250], order[:100]])
mean, std = X[train_idx].mean(0), X[train_idx].std(0) + 1e-6
norm = lambda a: (a - mean) / std  # noqa: E731

Xtr, ytr = norm(X[train_idx]), y[train_idx]
Xva, yva = norm(X[val_idx]), y[val_idx]

model = nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 10))
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-2)

for epoch in range(1, 31):
    model.train()
    perm = torch.randperm(len(train_idx))
    for i in range(0, len(train_idx), 32):
        b = perm[i:i + 32]
        opt.zero_grad()
        loss_fn(model(Xtr[b]), ytr[b]).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        train_acc = common.accuracy(model(Xtr), ytr)
        val_acc = common.accuracy(model(Xva), yva)
    common.log_metrics(epoch=epoch, train_acc=train_acc, val_acc=val_acc)

common.finalize(lambda raw: model(norm(raw)), train_idx, val_idx)
