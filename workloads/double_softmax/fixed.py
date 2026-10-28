import torch
import torch.nn as nn

import common

SEED = 0
common.set_determinism(SEED)
X, y, order = common.load_pool(SEED)
tr, va = order[:1000], order[1000:1300]
mean, std = X[tr].mean(0), X[tr].std(0) + 1e-6
norm = lambda a: (a - mean) / std  # noqa: E731

model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 10))
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1, 13):
    model.train()
    perm = torch.randperm(len(tr))
    total = 0.0
    for i in range(0, len(tr), 64):
        b = torch.as_tensor(tr)[perm[i:i + 64]]
        opt.zero_grad()
        loss = loss_fn(model(norm(X[b])), y[b])
        loss.backward()
        opt.step()
        total += loss.item() * len(b)
    model.eval()
    with torch.no_grad():
        val_acc = common.accuracy(model(norm(X[va])), y[va])
    common.log_metrics(epoch=epoch, train_loss=total / len(tr), val_acc=val_acc)

common.finalize(lambda raw: model(norm(raw)))
