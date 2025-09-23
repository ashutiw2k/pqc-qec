import torch
import torch.nn as nn
import math

# ---------------- Positional encoding (batch_first) ----------------
class PositionalEncodingBF(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)                                  # [L, C]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # [L, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, L, C] for batch_first broadcasting
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):  # x: [B, L, C]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ---------------- Simple Transformer Encoder -> angle regressor ----------------
class SimpleTransformer(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 out_shape: tuple,   # (K, Q, P)
                 d_model: int = 128,
                 nhead: int = 2,
                 num_encoder_layers: int = 2,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1,
                 pad_id: int = 0,
                 max_len: int = 128):
        super().__init__()
        self.K, self.Q, self.P = out_shape
        self.out_dim = self.K * self.Q * self.P
        self.d_model = d_model
        self.pad_id = pad_id

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoder = PositionalEncodingBF(d_model, dropout, max_len=max_len)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

        self.regression_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, self.out_dim)
        )

    def forward(self, input_ids, attention_mask=None):  # [B, L] or [L]
        # Check if input is batched or singular
        is_batched = input_ids.dim() == 2

        if not is_batched:
            input_ids = input_ids.unsqueeze(0)  # Add batch dimension for singular data
            if attention_mask is not None:
                attention_mask = attention_mask.unsqueeze(0)

        # Embedding + scale
        x = self.embedding(input_ids) * math.sqrt(self.d_model)  # [B, L, C]
        x = self.pos_encoder(x)

        # Key padding mask: True where padding should be ignored
        if attention_mask is None:
            key_padding_mask = (input_ids == self.pad_id)  # [B, L]
        else:
            key_padding_mask = ~attention_mask.bool()  # Invert mask to match padding logic

        # Encoder
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)  # [B, L, C]

        # Masked mean pool over tokens
        mask = (~key_padding_mask).unsqueeze(-1).float()  # [B, L, 1]
        denom = mask.sum(dim=1).clamp_min(1.0)            # [B, 1]
        pooled = (x * mask).sum(dim=1) / denom            # [B, C]

        # Regress angles
        y = self.regression_head(pooled)                  # [B, K*Q*P]
        y = y.view(-1, self.K, self.Q, self.P)            # [B, K, Q, P]

        if not is_batched:
            y = y.squeeze(0)  # Remove batch dimension for singular data

        return y
