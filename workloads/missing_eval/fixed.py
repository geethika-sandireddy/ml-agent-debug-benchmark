import common
import torch
from torch import nn

SEED = common.get_seed()
common.set_determinism(SEED)
X, y, order = common.load_pool(SEED)
tr, va = order[:1000], order[1000:1300]
mean, std = X[tr].mean(0), X[tr].std(0) + 1e-6
norm = lambda a: (a - mean) / std

model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.7), nn.Linear(128, 10))
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-2)

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
