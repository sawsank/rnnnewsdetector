from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def tokenize(text: str) -> List[str]:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


@dataclass(frozen=True)
class Vocab:
    stoi: dict[str, int]
    itos: list[str]
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"

    @property
    def pad_id(self) -> int:
        return self.stoi[self.pad_token]

    @property
    def unk_id(self) -> int:
        return self.stoi[self.unk_token]

    def encode(self, tokens: Sequence[str]) -> List[int]:
        unk = self.unk_id
        return [self.stoi.get(t, unk) for t in tokens]

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "itos": self.itos,
                    "pad_token": self.pad_token,
                    "unk_token": self.unk_token,
                },
                indent=2,
            )
        )

    @staticmethod
    def load_json(path: str | Path) -> "Vocab":
        obj = json.loads(Path(path).read_text())
        itos = list(obj["itos"])
        stoi = {t: i for i, t in enumerate(itos)}
        return Vocab(
            stoi=stoi,
            itos=itos,
            pad_token=obj.get("pad_token", "<pad>"),
            unk_token=obj.get("unk_token", "<unk>"),
        )


def build_vocab(
    token_seqs: Iterable[Sequence[str]],
    *,
    min_freq: int = 2,
    max_size: int = 50000,
    pad_token: str = "<pad>",
    unk_token: str = "<unk>",
) -> Vocab:
    freq: dict[str, int] = {}
    for seq in token_seqs:
        for t in seq:
            freq[t] = freq.get(t, 0) + 1

    tokens = [t for t, c in freq.items() if c >= min_freq]
    tokens.sort(key=lambda t: (-freq[t], t))
    tokens = tokens[: max(0, max_size - 2)]

    itos = [pad_token, unk_token, *tokens]
    stoi = {t: i for i, t in enumerate(itos)}
    return Vocab(stoi=stoi, itos=itos, pad_token=pad_token, unk_token=unk_token)


def load_true_fake_csvs(true_csv: str | Path, fake_csv: str | Path) -> pd.DataFrame:
    true_df = pd.read_csv(true_csv)
    fake_df = pd.read_csv(fake_csv)

    true_df = true_df.copy()
    fake_df = fake_df.copy()
    true_df["label"] = 1
    fake_df["label"] = 0

    df = pd.concat([true_df, fake_df], ignore_index=True)
    for col in ["title", "text"]:
        if col not in df.columns:
            df[col] = ""

    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")
    df["input_text"] = (df["title"].astype(str) + " " + df["text"].astype(str)).str.strip()
    df = df[["input_text", "label"]]
    return df


def make_splits(
    df: pd.DataFrame,
    *,
    val_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        df,
        test_size=val_size + test_size,
        random_state=seed,
        stratify=df["label"],
    )
    rel_test = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=rel_test,
        random_state=seed,
        stratify=temp_df["label"],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


class TextClsDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        vocab: Vocab,
        *,
        max_len: int = 400,
    ) -> None:
        self._texts = df["input_text"].astype(str).tolist()
        self._labels = df["label"].astype(int).tolist()
        self._vocab = vocab
        self._max_len = int(max_len)

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        tokens = tokenize(self._texts[idx])
        ids = self._vocab.encode(tokens)[: self._max_len]
        x = torch.tensor(ids, dtype=torch.long)
        y = int(self._labels[idx])
        return x, y


def collate_batch(
    batch: Sequence[Tuple[torch.Tensor, int]],
    *,
    pad_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xs, ys = zip(*batch)
    lengths = torch.tensor([x.numel() for x in xs], dtype=torch.long)
    max_len = int(lengths.max().item()) if len(xs) else 0
    x_padded = torch.full((len(xs), max_len), fill_value=int(pad_id), dtype=torch.long)
    for i, x in enumerate(xs):
        if x.numel() > 0:
            x_padded[i, : x.numel()] = x
    y = torch.tensor(ys, dtype=torch.float32)
    return x_padded, lengths, y

