# train_lm.py
import argparse, random, numpy as np, torch
import torch.nn.functional as F
from tokens import VOCAB, load_demo_tokens
from model import TinyTransformerLM

def batches(data, B=16, T=128):
    buf = []
    for seq in data:
        if len(seq) < 4: continue
        for i in range(0, len(seq)-1, 3):
            x = seq[max(0,i-T):i]
            y = seq[max(0,i-T)+1:i+1]
            if len(x)<T:
                pad = [0]*(T-len(x))
                x = pad + x
                y = pad + y
            buf.append((x,y))
    random.shuffle(buf)
    for i in range(0, len(buf), B):
        X = torch.tensor([x for x,_ in buf[i:i+B]], dtype=torch.long)
        Y = torch.tensor([y for _,y in buf[i:i+B]], dtype=torch.long)
        yield X, Y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--save_path", type=str, default="baseline.pt")
    args = ap.parse_args()

    data = load_demo_tokens()
    model = TinyTransformerLM(len(VOCAB), d=512, n=6, h=8, max_len=128)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for ep in range(args.epochs):
        total = 0.0; steps = 0
        for X, Y in batches(data, B=16, T=64):
            logits = model(X)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=0)
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.item()); steps += 1
        print(f"epoch {ep+1}: loss={total/max(1,steps):.4f}")
    torch.save(model.state_dict(), args.save_path)
    print("saved:", args.save_path)

if __name__ == "__main__":
    main()
