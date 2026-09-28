import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from preprocess import load_data, MAX_LEN
from model import Encoder, Decoder, BahdanauAttention


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"


# --------------------------------------------------
# LIGHTER TRAINING SETTINGS
# --------------------------------------------------

EMBEDDING_DIM = 64
UNITS = 128
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001

PAD = 0
SOS = 1
EOS = 2


# --------------------------------------------------
# LOSS FUNCTION
# --------------------------------------------------

def masked_loss(real, pred):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction="none"
    )

    loss = loss_object(real, pred)

    # Ignore <PAD> tokens
    mask = tf.cast(tf.not_equal(real, PAD), loss.dtype)

    return tf.reduce_sum(loss * mask) / (
        tf.reduce_sum(mask) + 1e-7
    )


# --------------------------------------------------
# TRAINING FUNCTION
# --------------------------------------------------

def train():

    print("Loading dataset...")

    X, Y, word2id, id2word = load_data()

    # Use only 3000 pairs for lightweight training
    X = X[:3000]
    Y = Y[:3000]

    vocab_size = len(word2id)

    print("Dataset ready.")
    print("Training pairs:", len(X))
    print("Vocabulary size:", vocab_size)

    # --------------------------------------------------
    # DATASET
    # --------------------------------------------------

    dataset = (
        tf.data.Dataset
        .from_tensor_slices((X, Y))
        .shuffle(len(X), reshuffle_each_iteration=True)
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(1)
    )

    # --------------------------------------------------
    # MODEL
    # --------------------------------------------------

    print("Building model...")

    encoder = Encoder(
        vocab_size,
        EMBEDDING_DIM,
        UNITS
    )

    attention = BahdanauAttention(UNITS)

    decoder = Decoder(
        vocab_size,
        EMBEDDING_DIM,
        UNITS,
        attention
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    )

    # --------------------------------------------------
    # TRAINING STEP
    # --------------------------------------------------

    @tf.function
    def train_step(inp, targ):

        loss = tf.constant(0.0, dtype=tf.float32)

        with tf.GradientTape() as tape:

            # Encoder
            enc_out, enc_h, enc_c = encoder(
                inp,
                training=True
            )

            hidden = enc_h
            cell = enc_c

            # Mask padding in encoder input
            enc_mask = tf.not_equal(inp, PAD)

            # First decoder input = <SOS>
            dec_input = tf.expand_dims(
                targ[:, 0],
                1
            )

            # Teacher forcing
            for t in range(1, MAX_LEN):

                predictions, hidden, cell, _ = decoder(
                    dec_input,
                    hidden,
                    cell,
                    enc_out,
                    enc_mask,
                    training=True
                )

                step_pred = predictions[:, 0, :]

                loss += masked_loss(
                    targ[:, t],
                    step_pred
                )

                # Teacher forcing:
                # use actual target word as next input
                dec_input = tf.expand_dims(
                    targ[:, t],
                    1
                )

            loss /= tf.cast(
                MAX_LEN - 1,
                tf.float32
            )

        # Model parameters
        variables = (
            encoder.trainable_variables
            + decoder.trainable_variables
        )

        gradients = tape.gradient(
            loss,
            variables
        )

        optimizer.apply_gradients(
            zip(gradients, variables)
        )

        return loss

    # --------------------------------------------------
    # TRAINING
    # --------------------------------------------------

    history = []

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("Starting training...")
    print("----------------------------------")

    for epoch in range(1, EPOCHS + 1):

        batch_losses = []

        for inp, targ in dataset:

            batch_loss = train_step(
                inp,
                targ
            )

            batch_losses.append(
                float(batch_loss.numpy())
            )

        epoch_loss = float(
            np.mean(batch_losses)
        )

        history.append(epoch_loss)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} "
            f"- loss: {epoch_loss:.4f}"
        )

    # --------------------------------------------------
    # SAVE MODEL WEIGHTS
    # --------------------------------------------------

    print()
    print("Saving model...")

    encoder.save_weights(
        MODEL_DIR / "encoder.weights.h5"
    )

    decoder.save_weights(
        MODEL_DIR / "decoder.weights.h5"
    )

    # --------------------------------------------------
    # SAVE CONFIGURATION
    # --------------------------------------------------

    config = {
        "embedding_dim": EMBEDDING_DIM,
        "units": UNITS,
        "max_len": MAX_LEN,
        "vocab_size": vocab_size
    }

    (MODEL_DIR / "config.json").write_text(
        json.dumps(
            config,
            indent=2
        ),
        encoding="utf-8"
    )

    # --------------------------------------------------
    # SAVE LOSS GRAPH
    # --------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        range(1, EPOCHS + 1),
        history,
        marker="o"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Seq2Seq Training Loss")

    plt.grid(
        True,
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "loss_curve.png",
        dpi=150
    )

    plt.close()

    # --------------------------------------------------
    # TRAINING COMPLETE
    # --------------------------------------------------

    print()
    print("----------------------------------")
    print("Training complete!")
    print("----------------------------------")
    print("Saved files:")
    print("1. models/encoder.weights.h5")
    print("2. models/decoder.weights.h5")
    print("3. models/config.json")
    print("4. outputs/loss_curve.png")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":
    train()