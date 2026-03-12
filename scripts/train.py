from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rnn_news.data import (
    TextClsDataset,
    Vocab,
    build_vocab,
    collate_batch,
    load_true_fake_csvs,
    make_splits,
    tokenize,
)
from rnn_news.model import RNNClassifier
from rnn_news.training import evaluate, load_checkpoint, save_checkpoint, set_seed, train_one_epoch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--true_csv", type=str, required=True)
    p.add_argument("--fake_csv", type=str, required=True)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_size", type=float, default=0.1)
    p.add_argument("--test_size", type=float, default=0.1)

    p.add_argument("--min_freq", type=int, default=2)
    p.add_argument("--max_vocab", type=int, default=50000)
    p.add_argument("--max_len", type=int, default=400)

    p.add_argument("--model", type=str, default="gru", choices=["gru", "lstm"])
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--bidirectional", action="store_true")
    p.add_argument("--no_bidirectional", action="store_true")

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])

    p.add_argument("--artifacts_dir", type=str, default="artifacts")
    p.add_argument("--checkpoints_dir", type=str, default="checkpoints")

    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--save_last", action="store_true", help="Save last epoch checkpoint to checkpoints/last.pt")
    p.add_argument("--patience", type=int, default=3, help="Early stop patience on validation accuracy")
    p.add_argument("--min_delta", type=float, default=0.0, help="Minimum val accuracy improvement to reset patience")
    return p.parse_args()


def choose_device(device_flag: str) -> torch.device:
    if device_flag == "cpu":
        return torch.device("cpu")
    if device_flag == "cuda":
        return torch.device("cuda")
    if device_flag == "mps":
        return torch.device("mps")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    artifacts_dir = Path(args.artifacts_dir)
    checkpoints_dir = Path(args.checkpoints_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    print(f"Using device: {device}")

    if args.eval_only:
        if not args.checkpoint:
            raise SystemExit("--eval_only requires --checkpoint")
        ckpt = load_checkpoint(args.checkpoint, map_location=device)
        vocab = Vocab.load_json(ckpt["vocab_path"])
        cfg = ckpt["config"]

        df = load_true_fake_csvs(args.true_csv, args.fake_csv)
        train_df, val_df, test_df = make_splits(df, val_size=args.val_size, test_size=args.test_size, seed=args.seed)
        test_ds = TextClsDataset(test_df, vocab, max_len=int(cfg["max_len"]))
        test_loader = DataLoader(
            test_ds,
            batch_size=int(cfg["batch_size"]),
            shuffle=False,
            collate_fn=lambda b: collate_batch(b, pad_id=vocab.pad_id),
        )

        model = RNNClassifier(
            vocab_size=len(vocab.itos),
            embed_dim=int(cfg["embed_dim"]),
            hidden_dim=int(cfg["hidden_dim"]),
            num_layers=int(cfg["num_layers"]),
            dropout=float(cfg["dropout"]),
            bidirectional=bool(cfg["bidirectional"]),
            rnn_type=str(cfg["model"]),
            pad_id=vocab.pad_id,
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        res = evaluate(model, test_loader, device)
        print(f"TEST  loss={res.loss:.4f}  acc={res.accuracy:.4f}")
        return

    df = load_true_fake_csvs(args.true_csv, args.fake_csv)
    train_df, val_df, test_df = make_splits(df, val_size=args.val_size, test_size=args.test_size, seed=args.seed)

    train_tokens = (tokenize(t) for t in train_df["input_text"].astype(str).tolist())
    vocab = build_vocab(train_tokens, min_freq=args.min_freq, max_size=args.max_vocab)
    vocab_path = artifacts_dir / "vocab.json"
    vocab.save_json(vocab_path)

    train_ds = TextClsDataset(train_df, vocab, max_len=args.max_len)
    val_ds = TextClsDataset(val_df, vocab, max_len=args.max_len)
    test_ds = TextClsDataset(test_df, vocab, max_len=args.max_len)

    collate = lambda b: collate_batch(b, pad_id=vocab.pad_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    bidirectional = True
    if args.no_bidirectional:
        bidirectional = False
    elif args.bidirectional:
        bidirectional = True

    model = RNNClassifier(
        vocab_size=len(vocab.itos),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=bidirectional,
        rnn_type=args.model,
        pad_id=vocab.pad_id,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val_acc = -1.0
    best_path = checkpoints_dir / "best.pt"
    last_path = checkpoints_dir / "last.pt"
    patience_counter = 0
    cfg = {
        "model": args.model,
        "embed_dim": args.embed_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "bidirectional": bidirectional,
        "batch_size": args.batch_size,
        "max_len": args.max_len,
    }

    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, device, grad_clip=args.grad_clip)
        va = evaluate(model, val_loader, device)
        print(f"E{epoch:02d}  train loss={tr.loss:.4f} acc={tr.accuracy:.4f} | val loss={va.loss:.4f} acc={va.accuracy:.4f}")

        if args.save_last:
            save_checkpoint(
                last_path,
                model=model,
                optimizer=optimizer,
                vocab_path=vocab_path,
                config=cfg,
            )

        if va.accuracy > best_val_acc + args.min_delta:
            best_val_acc = va.accuracy
            patience_counter = 0
            save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                vocab_path=vocab_path,
                config=cfg,
            )
        else:
            patience_counter += 1
            if args.patience > 0 and patience_counter >= args.patience:
                print(f"Early stopping triggered (patience={args.patience}).")
                break

    ckpt = load_checkpoint(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    te = evaluate(model, test_loader, device)
    print(f"TEST  loss={te.loss:.4f}  acc={te.accuracy:.4f}")
    print(f"Saved best checkpoint to: {best_path}")


if __name__ == "__main__":
    main()

