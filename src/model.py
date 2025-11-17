# model.py
import torch, torch.nn as nn

class TinyTransformerLM(nn.Module):
    def __init__(self, vocab_size, d=256, n=4, h=8, max_len=512):
        super().__init__()
        self.max_len = max_len
        self.tok = nn.Embedding(vocab_size, d)
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerDecoderLayer(
            d_model=d,
            nhead=h,
            dim_feedforward=4 * d,
            batch_first=True,
        )
        self.dec = nn.TransformerDecoder(layer, num_layers=n)
        self.lm_head = nn.Linear(d, vocab_size)

    def forward(self, x):
        """Forward pass.

        x: LongTensor of shape [B, T] with token ids.
        If T > self.max_len, we keep only the last self.max_len tokens
        to stay within the positional embedding range.
        """
        B, T = x.shape

        # If sequence is longer than max_len, use a sliding window
        if T > self.max_len:
            x = x[:, -self.max_len:]
            T = self.max_len

        # Positional indices [0..T-1]
        pos = torch.arange(T, device=x.device).unsqueeze(0)  # [1, T]

        # Token + positional embedding
        h = self.tok(x) + self.pos(pos)  # [B, T, d]

        # Causal (upper-triangular) attention mask
        attn_mask = torch.triu(
            torch.ones(T, T, device=x.device), diagonal=1
        ).bool()  # [T, T]

        # Transformer decoder (we don't use an encoder, so pass zeros as memory)
        out = self.dec(h, torch.zeros_like(h), tgt_mask=attn_mask)  # [B, T, d]

        # Project to vocabulary logits
        logits = self.lm_head(out)  # [B, T, vocab_size]
        return logits
