from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rnn_news.data import Vocab, tokenize  # noqa: E402
from rnn_news.model import RNNClassifier  # noqa: E402
from rnn_news.training import load_checkpoint  # noqa: E402


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@st.cache_resource
def load_model(checkpoint_path: str) -> tuple[RNNClassifier, Vocab, dict]:
    device = choose_device()
    ckpt = load_checkpoint(checkpoint_path, map_location=device)
    cfg = dict(ckpt["config"])
    vocab = Vocab.load_json(ckpt["vocab_path"])

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
    model.eval()
    return model, vocab, cfg


def encode_text(vocab: Vocab, text: str, *, max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = tokenize(text)
    ids = vocab.encode(tokens)[: int(max_len)]
    if len(ids) == 0:
        ids = [vocab.unk_id]
    x = torch.tensor([ids], dtype=torch.long)
    lengths = torch.tensor([len(ids)], dtype=torch.long)
    return x, lengths


def main() -> None:
    st.set_page_config(page_title="Fake News Detector (RNN)", page_icon="📰", layout="centered")
    st.title("Fake News Detector (RNN)")
    st.caption("GRU/LSTM text classifier trained on `News _dataset/True.csv` + `Fake.csv`.")

    default_ckpt = str((PROJECT_ROOT / "checkpoints" / "best.pt").resolve())
    ckpt_path = st.text_input("Checkpoint path", value=default_ckpt)

    col1, col2 = st.columns(2)
    with col1:
        title = st.text_input("Title", value="")
    with col2:
        max_len = st.number_input("Max tokens", min_value=20, max_value=800, value=200, step=20)

    body = st.text_area("Article text", value="", height=220, placeholder="Paste the article text here…")

    if st.button("Predict", type="primary"):
        if not Path(ckpt_path).exists():
            st.error(f"Checkpoint not found: {ckpt_path}")
            return

        model, vocab, cfg = load_model(ckpt_path)
        device = next(model.parameters()).device

        combined = (title + " " + body).strip()
        if not combined:
            st.warning("Please enter a title and/or article text.")
            return

        x, lengths = encode_text(vocab, combined, max_len=int(max_len))
        x = x.to(device)
        lengths = lengths.to(device)

        with torch.no_grad():
            logits = model(x, lengths)
            prob_true = float(torch.sigmoid(logits).item())

        prob_fake = 1.0 - prob_true
        pred_label = "TRUE" if prob_true >= 0.5 else "FAKE"

        st.subheader(f"Prediction: {pred_label}")
        st.write(f"Probability TRUE: **{prob_true:.3f}**")
        st.write(f"Probability FAKE: **{prob_fake:.3f}**")
        st.caption(f"Model: {cfg.get('model')} | bidirectional={cfg.get('bidirectional')} | device={device}")


if __name__ == "__main__":
    main()

