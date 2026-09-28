# Transformer for Humanitarian Tweet Classification

This project uses an **encoder-only Transformer** to classify humanitarian tweets. It is based on the self-attention architecture introduced in [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

The model is implemented in **PyTorch** and trained on the **HumAID** dataset. A custom **Byte Pair Encoding (BPE)** tokenizer is trained on the training split using Hugging Face `tokenizers`.

## Features

- Custom BPE tokenizer with a 16,000-token vocabulary
- Encoder-only Transformer classifier
- Sinusoidal positional encoding
- 6 Transformer encoder layers
- 8 attention heads
- 512-dimensional embeddings
- Adam optimizer with Transformer warm-up and learning-rate schedule.
- Validation-based early stopping
- Training and evaluation metric logging

## Project Structure

```text
.
├── data/
│   ├── dataset.py
│   └── dataloader.py
│
├── models/
│   └── model.py
│
├── tokenizer/
│   └── tokenizer.ipynb
│
├── training/
│   ├── train.py
│   └── evaluate.py
│
├── results/
│   ├── train_metrics.csv
│   └── evaluation_metrics.csv
│
├── utils.py
└── main.py
```

## Model Pipeline

```text
Tweet
  ↓
Text Cleaning
  ↓
BPE Tokenization
  ↓
Token + Positional Embeddings
  ↓
Transformer Encoder
  ↓
<SOS> Representation
  ↓
Linear Classification Layer
  ↓
Humanitarian Class
```

## Training

The model uses cross-entropy loss and the Adam configuration and learning-rate schedule described in *Attention Is All You Need*. Validation loss is monitored during training, with early stopping used to retain the best-performing checkpoint.

## Dataset

The project uses the **QCRI HumAID** humanitarian tweet dataset for training, validation, and testing.