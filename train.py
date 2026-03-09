import argparse
import os

import numpy as np
import torch
import tqdm

from model import GPTLanguageModel, device, n_embd, n_head, n_layer, dropout, block_size
from tokenizer import Tokenizer

parser = argparse.ArgumentParser(description="Train TinyThinker")
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--max-iters", type=int, default=30000)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--eval-interval", type=int, default=100)
parser.add_argument("--save-interval", type=int, default=1000)
parser.add_argument("--grad-clip", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--corpus", type=str, default="corpus_int32.npy")
args = parser.parse_args()

print(f"Using device: {device}")
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

data_mm = np.load(args.corpus, mmap_mode='r')
data = torch.from_numpy(data_mm.astype(np.int64, copy=False))

tokenizer = Tokenizer()

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

assert len(train_data) > block_size, (
    f"train_data (size={len(train_data)}) must be larger than block_size ({block_size})"
)
assert len(val_data) > block_size, (
    f"val_data (size={len(val_data)}) must be larger than block_size ({block_size})"
)


def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (args.batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(args.eval_interval)
        for k in range(args.eval_interval):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


model = GPTLanguageModel(
    vocab_size=tokenizer.vocab_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    block_size=block_size,
    dropout=dropout,
)

if os.path.exists("model.pth"):
    print("model.pth exists. Loading the model...")
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()
else:
    print("model.pth does not exist. Training from scratch.")

model = model.to(device)
model.train()

print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
os.makedirs("checkpoints", exist_ok=True)

for iter in tqdm.tqdm(range(args.max_iters)):
    if iter % args.eval_interval == 0 or iter == args.max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    if iter % args.save_interval == 0:
        torch.save(model.state_dict(), f"checkpoints/model_epoch_{iter}.pth")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

torch.save(model.state_dict(), "model.pth")
print("Model saved successfully!")
