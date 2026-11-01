import common
import torch
from torch import nn

SEED = common.get_seed()
common.set_determinism(SEED)
X, y, order = common.load_pool(SEED)
tr, va = order[:1000], order[1000:1300]
mean, std = X[tr].mean(0), X[tr].std(0) + 1e-6

Xtr, ytr = (X[tr] - mean) / std, y[tr]
Xva, yva = X[va], y[va]

model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 10))
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-2)

for epoch in range(1, 13):
    model.train()
    perm = torch.randperm(len(tr))
    total = 0.0
    for i in range(0, len(tr), 64):
        b = perm[i:i + 64]
        opt.zero_grad()
        loss = loss_fn(model(Xtr[b]), ytr[b])
        loss.backward()
        opt.step()
        total += loss.item() * len(b)
    model.eval()
    with torch.no_grad():
        train_acc = common.accuracy(model(Xtr), ytr)
        val_acc = common.accuracy(model(Xva), yva)
    common.log_metrics(epoch=epoch, train_loss=total / len(tr), train_acc=train_acc, val_acc=val_acc)

common.finalize(lambda raw: model(raw))
