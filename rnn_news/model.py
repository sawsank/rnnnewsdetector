from __future__ import annotations

import torch
import torch.nn as nn


class RNNClassifier(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = True,
        rnn_type: str = "gru",
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)

        rnn_type = rnn_type.lower().strip()
        rnn_kwargs = dict(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        if rnn_type == "gru":
            self.rnn = nn.GRU(**rnn_kwargs)
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(**rnn_kwargs)
        else:
            raise ValueError(f"Unsupported rnn_type={rnn_type!r} (use 'gru' or 'lstm')")

        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, 1),
        )

        self._rnn_type = rnn_type
        self._bidirectional = bidirectional

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T) int64 token ids
        lengths: (B,) int64 true lengths
        returns logits: (B,)
        """
        emb = self.dropout(self.embedding(x))  # (B, T, E)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, h = self.rnn(packed)

        if self._rnn_type == "lstm":
            h = h[0]  # (num_layers * num_directions, B, H)

        if self._bidirectional:
            last = torch.cat([h[-2], h[-1]], dim=-1)  # (B, 2H)
        else:
            last = h[-1]  # (B, H)

        logits = self.head(last).squeeze(-1)
        return logits

