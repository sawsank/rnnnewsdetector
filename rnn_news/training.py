from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class EpochResult:
    loss: float
    accuracy: float


def _accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).to(torch.float32)
    return float((preds == y).to(torch.float32).mean().item())


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> EpochResult:
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for x, lengths, y in loader:
        x = x.to(device)
        lengths = lengths.to(device)
        y = y.to(device)
        logits = model(x, lengths)
        loss = loss_fn(logits, y)
        total_loss += float(loss.item())
        total_acc += _accuracy_from_logits(logits, y)
        n_batches += 1

    if n_batches == 0:
        return EpochResult(loss=math.nan, accuracy=math.nan)
    return EpochResult(loss=total_loss / n_batches, accuracy=total_acc / n_batches)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    grad_clip: Optional[float] = 1.0,
) -> EpochResult:
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for x, lengths, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device)
        lengths = lengths.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x, lengths)
        loss = loss_fn(logits, y)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        optimizer.step()

        total_loss += float(loss.item())
        total_acc += _accuracy_from_logits(logits.detach(), y)
        n_batches += 1

    if n_batches == 0:
        return EpochResult(loss=math.nan, accuracy=math.nan)
    return EpochResult(loss=total_loss / n_batches, accuracy=total_acc / n_batches)


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    vocab_path: str | Path,
    config: Dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "vocab_path": str(vocab_path),
            "config": config,
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> Dict:
    return torch.load(Path(path), map_location=map_location)

