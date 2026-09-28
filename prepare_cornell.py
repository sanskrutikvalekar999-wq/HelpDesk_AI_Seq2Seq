import ast
import io
import re
import zipfile
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ZIP_PATH = ROOT / "cornell_movie_dialogs_corpus.zip"
OUT_PATH = DATA_DIR / "cornell_pairs.csv"

# Official Cornell legacy download URL commonly used for this corpus.
URL = "https://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip"

MAX_WORDS = 15


def clean_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9!?',. ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def word_count(text: str) -> int:
    return len(text.split())


def download_if_needed():
    if ZIP_PATH.exists():
        print(f"Using existing archive: {ZIP_PATH}")
        return

    print("Downloading Cornell Movie-Dialogs Corpus...")
    try:
        response = requests.get(URL, timeout=60)
        response.raise_for_status()
        ZIP_PATH.write_bytes(response.content)
        print(f"Saved: {ZIP_PATH}")
    except Exception as exc:
        print("\nAutomatic download failed.")
        print("Download the corpus manually from:")
        print("https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html")
        print("Save the ZIP as:")
        print(ZIP_PATH)
        raise SystemExit(f"Reason: {exc}")


def locate_file(zf, filename):
    for name in zf.namelist():
        if name.lower().endswith(filename.lower()):
            return name
    raise FileNotFoundError(filename)


def parse_cornell():
    download_if_needed()
    DATA_DIR.mkdir(exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        lines_name = locate_file(zf, "movie_lines.txt")
        conv_name = locate_file(zf, "movie_conversations.txt")

        # Cornell files are encoded with cp1252 in many distributions.
        raw_lines = zf.read(lines_name).decode("iso-8859-1", errors="replace").splitlines()
        raw_convs = zf.read(conv_name).decode("iso-8859-1", errors="replace").splitlines()

    utterances = {}
    for row in raw_lines:
        parts = row.split(" +++$+++ ")
        if len(parts) >= 5:
            line_id = parts[0].strip()
            utterances[line_id] = parts[4].strip()

    pairs = []
    for row in raw_convs:
        parts = row.split(" +++$+++ ")
        if len(parts) < 4:
            continue

        try:
            line_ids = ast.literal_eval(parts[3].strip())
        except (ValueError, SyntaxError):
            continue

        for i in range(len(line_ids) - 1):
            q = clean_text(utterances.get(line_ids[i], ""))
            a = clean_text(utterances.get(line_ids[i + 1], ""))

            if not q or not a:
                continue
            if not (1 <= word_count(q) <= MAX_WORDS):
                continue
            if not (1 <= word_count(a) <= MAX_WORDS):
                continue

            pairs.append((q, a))

    df = pd.DataFrame(pairs, columns=["question", "answer"])
    df = df.drop_duplicates().reset_index(drop=True)

    # Keep the file manageable and reproducible while retaining a large training pool.
    df.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"Created {OUT_PATH}")
    print(f"Pairs: {len(df):,}")


if __name__ == "__main__":
    parse_cornell()
