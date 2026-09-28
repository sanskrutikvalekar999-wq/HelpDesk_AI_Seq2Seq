# HelpDesk AI – Seq2Seq Chatbot with Bahdanau Attention

## Project basis
This project follows the supplied project brief:
- Seq2Seq architecture
- LSTM Encoder
- LSTM Decoder
- Bahdanau Attention
- Teacher Forcing
- Categorical cross-entropy with padding mask
- 50 training epochs (configurable)
- Loss curve
- Saved model weights
- Streamlit chat interface

## Dataset
For the generative Seq2Seq demonstration, this version uses the **Cornell Movie-Dialogs Corpus**, which is explicitly allowed by the project brief as an open-ended conversational dataset.

Important: Cornell contains fictional movie conversations, not customer-support conversations. Therefore, this project demonstrates the requested Seq2Seq chatbot architecture; it should not be described as a trained customer-support production bot. A real customer-support dataset can later replace the Cornell data while keeping the model pipeline.

Cornell official source:
https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html

The official corpus contains 220,579 conversational exchanges, 304,713 utterances, 10,292 character pairs, and 617 movies.

## Folder structure

```text
HelpDesk_AI_Seq2Seq_Cornell/
│
├── data/
│   ├── README_DATASET.md
│   └── sample_pairs.csv
├── models/
├── outputs/
├── notebooks/
│   └── HelpDesk_AI_Seq2Seq.ipynb
├── prepare_cornell.py
├── preprocess.py
├── model.py
├── train.py
├── chatbot.py
├── app.py
├── requirements.txt
├── PROJECT_REPORT.md
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Step 1 – Download and prepare Cornell

Run:

```bash
python prepare_cornell.py
```

The script downloads the official Cornell archive, extracts it, converts consecutive dialogue turns into question-answer pairs, removes unsuitable/empty rows, and creates:

```text
data/cornell_pairs.csv
```

It limits each input and response to 15 words, matching the project brief.

If automatic downloading is blocked on your computer, download the corpus manually from the official Cornell page and place the ZIP in the project folder as:

```text
cornell_movie_dialogs_corpus.zip
```

Then run:

```bash
python prepare_cornell.py
```

## Step 2 – Train

```bash
python train.py
```

The default training configuration uses:
- maximum sequence length: 15
- vocabulary limit: 12,000 words
- maximum training pairs: 20,000
- embedding size: 128
- LSTM units: 256
- batch size: 64
- epochs: 50

The maximum number of pairs is intentionally configurable so the project can run on a normal student computer.

Outputs:
- `models/encoder.weights.h5`
- `models/decoder.weights.h5`
- `models/vocab.json`
- `outputs/loss_curve.png`

## Step 3 – Run chatbot

```bash
streamlit run app.py
```

Then open the local Streamlit address shown in the terminal.

## Quick test without Streamlit

```bash
python chatbot.py
```

## Academic explanation

### Input
User enters a sentence.

### Encoder
The Encoder LSTM reads the input sequence and produces hidden/context information.

### Attention
Bahdanau Attention calculates which encoder outputs are important for the current decoder step.

### Decoder
The Decoder LSTM generates the answer one word at a time.

### Teacher Forcing
During training, the correct previous target word is supplied to the decoder instead of always using its own previous prediction.

### Inference
During testing:
1. Encode the user sentence.
2. Start with `<SOS>`.
3. Predict one word.
4. Feed the predicted word back to the decoder.
5. Continue until `<EOS>` or the maximum length.

## Important project limitation
A Seq2Seq model trained on movie dialogue can produce conversational responses, but it is not a domain-safe HelpDesk system. For real customer support, retrain the same architecture on genuine support question-answer conversations and add safety, fallback, and evaluation rules.
