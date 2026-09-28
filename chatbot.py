import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from model import Encoder, Decoder, BahdanauAttention
from preprocess import clean_text, encode_input, MAX_LEN

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
VOCAB_PATH = MODEL_DIR / "vocab.json"

PAD = "<PAD>"
SOS = "<SOS>"
EOS = "<EOS>"
UNK = "<UNK>"


class Chatbot:
    def __init__(self):
        if not VOCAB_PATH.exists():
            raise FileNotFoundError(
                "Model vocabulary not found. Run: python train.py"
            )

        vocab = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
        self.word2id = vocab["word2id"]
        self.id2word = {int(k): v for k, v in vocab["id2word"].items()}

        config = json.loads(
            (MODEL_DIR / "config.json").read_text(encoding="utf-8")
        )

        vocab_size = config["vocab_size"]
        embedding_dim = config["embedding_dim"]
        units = config["units"]

        self.encoder = Encoder(vocab_size, embedding_dim, units)
        self.attention = BahdanauAttention(units)
        self.decoder = Decoder(vocab_size, embedding_dim, units, self.attention)

        dummy = tf.constant([[self.word2id[PAD]]], dtype=tf.int32)
        self.encoder(dummy)
        self.decoder(
            tf.constant([[self.word2id[SOS]]], dtype=tf.int32),
            tf.zeros((1, units)),
            tf.zeros((1, units)),
            tf.zeros((1, MAX_LEN, units)),
            tf.ones((1, MAX_LEN), dtype=tf.bool),
        )

        self.encoder.load_weights(MODEL_DIR / "encoder.weights.h5")
        self.decoder.load_weights(MODEL_DIR / "decoder.weights.h5")

    def reply(self, text):
        text = clean_text(text)
        if not text:
            return "Please enter a message."

        inp = tf.convert_to_tensor(
            encode_input(text, self.word2id)[None, :], dtype=tf.int32
        )

        enc_out, hidden, cell = self.encoder(inp, training=False)
        enc_mask = tf.not_equal(inp, self.word2id[PAD])

        dec_input = tf.constant([[self.word2id[SOS]]], dtype=tf.int32)
        words = []

        for _ in range(MAX_LEN - 1):
            logits, hidden, cell, _ = self.decoder(
                dec_input,
                hidden,
                cell,
                enc_out,
                enc_mask,
                training=False,
            )
            next_id = int(tf.argmax(logits[0, 0]).numpy())

            if next_id == self.word2id[EOS]:
                break
            if next_id in (self.word2id[PAD], self.word2id[SOS]):
                break

            word = self.id2word.get(next_id, UNK)
            if word == UNK:
                break

            words.append(word)
            dec_input = tf.constant([[next_id]], dtype=tf.int32)

        return " ".join(words).strip() or "I am not sure how to respond to that."


if __name__ == "__main__":
    bot = Chatbot()
    print("Type 'exit' to stop.")
    while True:
        msg = input("You: ").strip()
        if msg.lower() == "exit":
            break
        print("Bot:", bot.reply(msg))
