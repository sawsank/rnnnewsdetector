# RNN Fake News Classifier (True vs Fake)

A lightweight example project for binary text classification using an RNN (GRU or LSTM) on a small news dataset.

- `News _dataset/True.csv` → label 1 (true news)
- `News _dataset/Fake.csv` → label 0 (fake news)

## Features

- Tokenization + vocabulary building
- RNN model (GRU/LSTM) in `rnn_news/model.py`
- Training loop in `scripts/train.py`
- Checkpointing and metrics

## Prerequisites

- Python 3.8+
- `virtualenv` or `python -m venv`

## Setup

```bash
cd /Users/shasankjoshi/Desktop/Antigravity/RNN
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

## Train

```bash
python scripts/train.py \
  --true_csv "News _dataset/True.csv" \
  --fake_csv "News _dataset/Fake.csv" \
  --model gru \
  --epochs 3 \
  --batch_size 64 \
  --save_last \
  --patience 3
```

### Checkpoint outputs

- `checkpoints/best.pt` (best validation model)
- `checkpoints/last.pt` (last epoch model, if `--save_last`)
- `artifacts/vocab.json` (saved vocabulary)

## Evaluate

```bash
python scripts/train.py \
  --true_csv "News _dataset/True.csv" \
  --fake_csv "News _dataset/Fake.csv" \
  --eval_only \
  --checkpoint checkpoints/best.pt
```

## Quick commands

```bash
# setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# train
python scripts/train.py \
  --true_csv "News _dataset/True.csv" \
  --fake_csv "News _dataset/Fake.csv" \
  --model gru \
  --epochs 3 \
  --batch_size 64 \
  --save_last \
  --patience 3

# evaluate
python scripts/train.py \
  --true_csv "News _dataset/True.csv" \
  --fake_csv "News _dataset/Fake.csv" \
  --eval_only \
  --checkpoint checkpoints/best.pt

# optional run
python scripts/train.py --help
```

## Notes

- Ensure `News _dataset/` path is correct (it contains a space in current name).
- Add any new checkpoints/artifacts to `.gitignore` (`artifacts/`, `checkpoints/`, `runs/`).

