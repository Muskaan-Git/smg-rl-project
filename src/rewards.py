# rewards.py
# General rewards for symbolic music (monophonic)
import numpy as np
import torch
from tokens import itos

def style_prior_reward(logits_seq, actions_seq):
    """Average log-prob under a *frozen* baseline LM over the taken actions."""
    # logits_seq: [T, V] torch tensor from frozen baseline
    # actions_seq: [T] token ids chosen by current policy
    logp = torch.log_softmax(logits_seq, dim=-1)
    idx = torch.arange(actions_seq.size(0))
    picked = logp[idx, actions_seq]
    # normalize to [0,1] using a rough linear transform
    # Note: for simplicity we map mean logp from [-8,0] => [0,1]
    m = picked.mean().item()
    s = (m + 8.0)/8.0
    return max(0.0, min(1.0, float(s)))

def density_reward(tokens, window=32, target=(10,24)):
    """Reward if number of NOTE_ON in sliding window stays within [lo, hi]."""
    ev = [itos.get(t,"") for t in tokens]
    on_idx = [1 if x.startswith("NOTE_ON_") else 0 for x in ev]
    if not on_idx: return 0.0
    lo, hi = target
    scores = []
    for i in range(0, len(on_idx), window):
        cnt = sum(on_idx[i:i+window])
        if cnt < lo: scores.append(cnt/lo)
        elif cnt > hi: scores.append(hi/max(hi, cnt))
        else: scores.append(1.0)
    return sum(scores)/len(scores)

def silence_reward(tokens, max_sil_steps=16):
    """Penalize long runs of TIME_SHIFT without NOTE_ON."""
    ev = [itos.get(t,"") for t in tokens]
    run = 0; worst = 0
    for e in ev:
        if e.startswith("TIME_SHIFT_"):
            run += int(e.split("_")[-1])
            worst = max(worst, run)
        elif e.startswith("NOTE_ON_"):
            run = 0
    return 1.0 if worst <= max_sil_steps else max(0.0, 1.0 - (worst-max_sil_steps)/max_sil_steps)

def repetition_reward(tokens, n=4):
    grams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    if not grams: return 1.0
    dup = len(grams) - len(set(grams))
    rate = dup/len(grams)
    return 1.0 - rate

def total_reward(tokens, style_score):
    # Weighted sum; tune as needed
    return 0.4*style_score + 0.25*density_reward(tokens) + 0.2*silence_reward(tokens) + 0.15*repetition_reward(tokens)
