# PROJECT REPORT
# HelpDesk AI: Chatbot using Seq2Seq Model with Attention Mechanism

## 1. Introduction

A chatbot is a software system that communicates with users using natural language. Traditional rule-based chatbots depend on fixed rules and predefined responses.

This project demonstrates a generative chatbot using a **Sequence-to-Sequence (Seq2Seq)** deep learning architecture. The system uses an **LSTM Encoder**, **Bahdanau Attention**, and an **LSTM Decoder** to generate a response word by word.

The project is based on the supplied project brief.

## 2. Problem Statement

The aim is to build a chatbot that can understand an input sentence and generate a natural-language response using a deep learning Seq2Seq model.

## 3. Objectives

1. Prepare conversational text data.
2. Clean and tokenize text.
3. Build a vocabulary.
4. Add special tokens `<SOS>`, `<EOS>`, `<PAD>`, and `<UNK>`.
5. Build an LSTM Encoder.
6. Implement Bahdanau Attention.
7. Build an LSTM Decoder.
8. Train using Teacher Forcing.
9. Use categorical cross-entropy with padding masking.
10. Generate responses word by word.
11. Plot the training loss.
12. Save model weights.
13. Provide a Streamlit chat interface.

## 4. Dataset

The project uses the **Cornell Movie-Dialogs Corpus** for the generative dialogue demonstration.

According to the official Cornell description, the corpus contains 220,579 conversational exchanges, 304,713 utterances, 10,292 character pairs, and 617 movies.

The data consists of fictional conversations extracted from movie scripts. Therefore, it is important to state that this dataset is **not a customer-support dataset**. It is used here to demonstrate the open-ended conversational Seq2Seq architecture permitted by the project brief.

## 5. Data Preprocessing

The following steps are performed:

### Step 1: Load dialogue data
The Cornell conversation and utterance files are read from the downloaded corpus.

### Step 2: Create question-answer pairs
For every conversation, one utterance is paired with the next utterance.

Example:

```text
Input  → Hello.
Target → Hi there.
```

### Step 3: Clean text
- Convert text to lowercase.
- Remove unwanted characters.
- Remove extra spaces.
- Remove empty examples.

### Step 4: Limit sequence length
The project brief specifies a maximum of about 10–15 words. This implementation uses a maximum of 15 words.

### Step 5: Special tokens

```text
<SOS> = Start of Sentence
<EOS> = End of Sentence
<PAD> = Padding
<UNK> = Unknown Word
```

## 6. Vocabulary

A vocabulary maps words to integer IDs.

Example:

```text
hello → 15
help  → 42
<PAD> → 0
<SOS> → 1
<EOS> → 2
<UNK> → 3
```

The model cannot directly process normal text, so the words are converted into integer sequences.

## 7. System Architecture

```text
              USER INPUT
                  |
                  v
          Text Preprocessing
                  |
                  v
          Tokenization + IDs
                  |
                  v
          +----------------+
          |  LSTM Encoder  |
          +----------------+
                  |
          Encoder Outputs
                  |
                  v
        +-------------------+
        | Bahdanau Attention|
        +-------------------+
                  |
                  v
          +----------------+
          |  LSTM Decoder  |
          +----------------+
                  |
             Word Prediction
                  |
                  v
          Response Sentence
```

## 8. Encoder

The Encoder reads the input sequence one word at a time.

It uses:
- Embedding layer
- LSTM layer
- Hidden state
- Cell state

The Encoder produces a sequence of hidden representations that are used by the Attention mechanism.

## 9. Attention Mechanism

A basic Encoder-Decoder model can compress the input into a fixed representation.

Attention improves this process by allowing the Decoder to focus on different Encoder outputs at different time steps.

### Bahdanau Attention

At each decoder step:
1. The current decoder hidden state is considered.
2. It is compared with encoder outputs.
3. Attention scores are calculated.
4. Softmax converts scores into attention weights.
5. A weighted context vector is created.

Simple flow:

```text
Encoder Outputs
      |
      v
Attention Scores
      |
      v
Attention Weights
      |
      v
Context Vector
      |
      v
Decoder
```

## 10. Decoder

The Decoder generates the response one word at a time.

It receives:
- Previous target word
- Previous hidden state
- Previous cell state
- Attention context

It predicts the next word.

## 11. Teacher Forcing

Teacher Forcing is used during training.

Instead of always feeding the Decoder's previous prediction back into the next step, the correct previous target word is supplied.

Example:

```text
Target:  I   am   fine   <EOS>

Decoder:
Step 1 → I
Step 2 → am
Step 3 → fine
Step 4 → <EOS>
```

This helps training converge more efficiently.

## 12. Loss Function

Sparse categorical cross-entropy is used for next-word prediction.

Padding tokens are masked so that `<PAD>` positions do not contribute to the training loss.

Conceptually:

```text
Loss = Cross-Entropy(actual word, predicted word)
```

Only meaningful target tokens contribute to the reported loss.

## 13. Training

The implementation is configured for approximately 50 epochs, as suggested in the project timeline.

Main settings:

| Parameter | Value |
|---|---:|
| Maximum sequence length | 15 |
| Vocabulary limit | 12,000 |
| Training pairs | 20,000 maximum |
| Embedding size | 128 |
| LSTM units | 256 |
| Batch size | 64 |
| Epochs | 50 |
| Optimizer | Adam |
| Learning rate | 0.001 |

The training-pair limit is configurable so that the project remains practical on student hardware.

## 14. Inference

During inference, the correct target sentence is not available.

The chatbot follows this loop:

```text
User Input
    ↓
Encoder
    ↓
<SOS>
    ↓
Predict next word
    ↓
Feed predicted word back
    ↓
Predict next word
    ↓
Repeat
    ↓
<EOS> or maximum length
```

## 15. Deployment

A Streamlit interface is included.

Run:

```bash
streamlit run app.py
```

The user can enter a message and receive a generated response.

## 16. Expected Output

Example format:

```text
You: hello
Bot: hi there

You: how are you
Bot: I am fine
```

Actual responses depend on the trained model and dataset.

## 17. Advantages

- Generates responses word by word.
- Uses deep learning rather than fixed rules.
- Attention helps the Decoder use relevant encoder information.
- Teacher Forcing supports effective training.
- Streamlit provides a simple user interface.
- The same architecture can be retrained on domain-specific support data.

## 18. Limitations

- Cornell dialogue is fictional movie conversation.
- It is not suitable by itself for real customer support.
- A small training subset can produce repetitive or incorrect responses.
- Seq2Seq models can generate grammatically imperfect responses.
- The model does not have external knowledge or a database.
- Production customer support would require a genuine support dataset, evaluation, fallback handling, and safety controls.

## 19. Future Scope

1. Replace Cornell data with genuine customer-support conversations.
2. Increase training data.
3. Tune embedding size and LSTM units.
4. Compare Bahdanau and Luong attention.
5. Add BLEU or other response-quality evaluation.
6. Add intent detection.
7. Add FAQ/database retrieval for factual questions.
8. Add confidence-based fallback responses.
9. Deploy the application online.

## 20. Conclusion

This project demonstrates how a Seq2Seq chatbot can generate conversational responses using an LSTM Encoder, Bahdanau Attention, and LSTM Decoder.

The complete pipeline includes data preprocessing, vocabulary creation, Teacher Forcing, masked loss calculation, training, inference, model saving, and a Streamlit interface.

The implementation therefore covers the major technical stages required by the supplied project brief while keeping the dataset limitation clearly documented.
