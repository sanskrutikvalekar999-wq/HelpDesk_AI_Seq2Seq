import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

PAD = "<PAD>"
SOS = "<SOS>"
EOS = "<EOS>"
UNK = "<UNK>"

SPECIAL_TOKENS = [PAD, SOS, EOS, UNK]

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "cornell_pairs.csv"
VOCAB_PATH = ROOT / "models" / "vocab.json"

MAX_LEN = 15
MAX_VOCAB = 12000
MAX_PAIRS = 20000


def clean_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9!?',. ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text):
    return clean_text(text).split()


def build_vocab(texts):
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))

    words = [w for w, _ in counter.most_common(MAX_VOCAB - len(SPECIAL_TOKENS))]
    word2id = {w: i for i, w in enumerate(SPECIAL_TOKENS)}
    for w in words:
        if w not in word2id:
            word2id[w] = len(word2id)

    id2word = {str(i): w for w, i in word2id.items()}
    return word2id, id2word


def encode_input(text, word2id):
    ids = [word2id.get(w, word2id[UNK]) for w in tokenize(text)[:MAX_LEN]]
    ids += [word2id[PAD]] * (MAX_LEN - len(ids))
    return np.array(ids, dtype=np.int32)


def encode_target(text, word2id):
    tokens = tokenize(text)[: MAX_LEN - 2]
    ids = [word2id[SOS]]
    ids += [word2id.get(w, word2id[UNK]) for w in tokens]
    ids += [word2id[EOS]]
    ids += [word2id[PAD]] * (MAX_LEN - len(ids))
    return np.array(ids[:MAX_LEN], dtype=np.int32)


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run: python prepare_cornell.py"
        )

    df = pd.read_csv(DATA_PATH).dropna(subset=["question", "answer"])
    df["question"] = df["question"].map(clean_text)
    df["answer"] = df["answer"].map(clean_text)

    df = df[
        df["question"].str.split().str.len().between(1, MAX_LEN)
        & df["answer"].str.split().str.len().between(1, MAX_LEN)
    ]
    df = df.drop_duplicates().head(MAX_PAIRS).reset_index(drop=True)

    word2id, id2word = build_vocab(
        pd.concat([df["question"], df["answer"]], ignore_index=True).tolist()
    )

    X = np.stack([encode_input(x, word2id) for x in df["question"]])
    Y = np.stack([encode_target(x, word2id) for x in df["answer"]])

    VOCAB_PATH.parent.mkdir(exist_ok=True)
    VOCAB_PATH.write_text(
        json.dumps(
            {
                "word2id": word2id,
                "id2word": id2word,
                "max_len": MAX_LEN,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return X, Y, word2id, id2word
