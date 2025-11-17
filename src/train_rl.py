# train_rl.py
# REINFORCE fine-tuning for symbolic music generation
# Uses a frozen baseline LM as a style prior.

import argparse
import torch
from torch.distributions.categorical import Categorical

from tokens import VOCAB, itos
from model import TinyTransformerLM
from rewards import total_reward, style_prior_reward


def sample_with_logits(model, T=128):
    """
    Roll out one sequence from the current policy model.
    Returns:
      traj: list[int] of token ids (actions)
      logits_seq: Tensor [T, V] of logits from the policy
    """
    model.eval()
    x = torch.zeros(1, 1, dtype=torch.long)  # start with PAD=0
    traj = []
    logits_seq = []

    for _ in range(T):
        logits = model(x)[:, -1, :]          # [1, V]
        dist = Categorical(logits=logits)
        a = dist.sample()                    # [1]
        token_id = int(a.item())

        traj.append(token_id)
        logits_seq.append(logits.squeeze(0)) # [V]

        x = torch.cat([x, a.view(1, 1)], dim=1)

        # Stop if EOS token generated
        if itos[token_id] == "EOS":
            break

    if len(logits_seq) > 0:
        logits_seq = torch.stack(logits_seq, dim=0)  # [T, V]
    else:
        logits_seq = torch.zeros(0, len(VOCAB))

    return traj, logits_seq


def reinforce_update(policy, opt, baseline_frozen, episodes=16, T=96):
    """
    One REINFORCE update loop over several sampled sequences.

    policy          : trainable TinyTransformerLM
    opt             : optimizer
    baseline_frozen : frozen LM for style prior reward
    episodes        : how many rollouts per RL epoch
    T               : max length of each rollout
    """
    policy.train()
    total_loss = 0.0
    used = 0

    for _ in range(episodes):
        traj, logits_seq_policy = sample_with_logits(policy, T=T)
        if len(traj) == 0:
            continue

        acts = torch.tensor(traj, dtype=torch.long)

        # ---- compute baseline logits on the SAME action sequence ----
        with torch.no_grad():
            x = torch.zeros(1, 1, dtype=torch.long)
            base_logits_list = []
            for a in traj:
                logits_base = baseline_frozen(x)[:, -1, :]  # [1, V]
                base_logits_list.append(logits_base.squeeze(0))
                x = torch.cat([x, torch.tensor([[a]])], dim=1)
            base_logits = torch.stack(base_logits_list, dim=0)  # [T, V]

        # ---- reward from rewards.py ----
        style_score = style_prior_reward(base_logits, acts)
        R = total_reward(traj, style_score)

        # ---- REINFORCE loss: -R * sum_t log π(a_t) ----
        logp_all = torch.log_softmax(logits_seq_policy, dim=-1)  # [T, V]
        idx = torch.arange(len(acts))
        logp = logp_all[idx, acts]                               # [T]

        loss = -(logp.sum()) * R

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += float(loss.item())
        used += 1

    if used == 0:
        return 0.0
    return total_loss / used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=str, required=True,
                    help="Path to baseline LM checkpoint (baseline.pt)")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--save_path", type=str, default="policy_rl.pt")
    args = ap.parse_args()

    # --------- LOAD BASELINE (BIG MODEL: d=512, n=6) ----------
    policy = TinyTransformerLM(len(VOCAB), d=512, n=6, h=8, max_len=128)
    sd = torch.load(args.baseline, map_location="cpu")
    policy.load_state_dict(sd)

    # Frozen copy used for style prior reward
    baseline_frozen = TinyTransformerLM(len(VOCAB), d=512, n=6, h=8, max_len=128)
    baseline_frozen.load_state_dict(sd)
    for p in baseline_frozen.parameters():
        p.requires_grad = False

    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    # --------- RL TRAINING LOOP ----------
    for ep in range(args.epochs):
        avg_loss = reinforce_update(policy, opt, baseline_frozen,
                                    episodes=16, T=96)
        print(f"RL epoch {ep+1}: reinforce_loss={avg_loss:.2f}")

    # --------- SAVE RL POLICY ----------
    torch.save(policy.state_dict(), args.save_path)
    print("saved:", args.save_path)


if __name__ == "__main__":
    main()
