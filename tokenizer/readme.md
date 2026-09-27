# BPE Tokenizer

This project uses a custom **Byte Pair Encoding (BPE)** tokenizer trained on the training split of the Humanitarian Tweet dataset.

## 1. Clean the Training Text

Before training the tokenizer, the tweets are cleaned to normalize the text and replace information that does not need to be represented directly.

```python
import re

def clean_text(text):
    text = " ".join(text.split())
    text = re.sub(r"http\S+|www\S+|https\S+", "<URL>", text)
    text = re.sub(r"@\w+", "<MENTION>", text)
    text = text.replace("&amp;", "&")
    return text

cleaned_training_texts = [
    clean_text(text) for text in ds["train"]["tweet_text"]
]
```

Only the **training split** is used to train the BPE tokenizer. The same cleaning function is later applied to the validation and test data.

## 2. Initialize the BPE Tokenizer

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE

tokenizer = Tokenizer(
    BPE(unk_token="<UNK>")
)
```

`<UNK>` represents tokens that cannot be represented using the learned vocabulary.

## 3. Configure Pre-Tokenization

```python
from tokenizers.pre_tokenizers import Whitespace

tokenizer.pre_tokenizer = Whitespace()
```

The whitespace pre-tokenizer first splits the input into word and punctuation units before BPE is applied.

For example:

```text
"Hello world!"
        ↓
["Hello", "world", "!"]
        ↓
BPE subword tokenization
```

BPE can then split less common words into smaller learned subword units.

## 4. Configure the BPE Trainer

```python
from tokenizers.trainers import BpeTrainer

tokenizer_trainer = BpeTrainer(
    vocab_size=16000,
    special_tokens=[
        "<PAD>",
        "<UNK>",
        "<SOS>",
        "<EOS>",
        "<URL>",
        "<MENTION>"
    ]
)
```

A vocabulary size of **16,000** is used. The tokenizer also reserves several special tokens:

- `<PAD>` — pads sequences to a common length.
- `<UNK>` — represents unknown tokens.
- `<SOS>` — marks the start of a sequence.
- `<EOS>` — marks the end of a sequence.
- `<URL>` — represents URLs removed during preprocessing.
- `<MENTION>` — represents user mentions removed during preprocessing.

**Note**: You can customize special tokens to your needs (data-specific tokens, domain-specific tokens, etc.).

## 5. Train the Tokenizer

```python
tokenizer.train_from_iterator(
    cleaned_training_texts,
    trainer=tokenizer_trainer
)
```

The BPE algorithm learns its vocabulary and merge rules from the cleaned **training tweets only**.

## Post-Training Vocabulary Configuration
### 6. Add Start and End Tokens

After training, a post-processor automatically adds `<SOS>` and `<EOS>` to every encoded tweet.

```python
from tokenizers.processors import TemplateProcessing

tokenizer.post_processor = TemplateProcessing(
    single="<SOS> $A <EOS>",
    special_tokens=[
        ("<SOS>", tokenizer.token_to_id("<SOS>")),
        ("<EOS>", tokenizer.token_to_id("<EOS>"))
    ]
)
```

Here, `$A` represents the tokenized tweet.

For example:

```text
["earthquake", "reported", "today"]
                 ↓
["<SOS>", "earthquake", "reported", "today", "<EOS>"]
```

**Note**: If you saved before adding the post-processor, you will need to add it manually after loading the tokenizer. Just call the same `TemplateProcessing` code after loading the tokenizer.

### 7. Configure Truncation and Padding

Sequences are limited to a maximum length of **128 tokens**.

```python
MAX_LENGTH = 128

tokenizer.enable_truncation(
    max_length=MAX_LENGTH
)

tokenizer.enable_padding(
    length=MAX_LENGTH,
    pad_id=tokenizer.token_to_id("<PAD>"),
    pad_token="<PAD>"
)
```

Long sequences are truncated to 128 tokens, while shorter sequences are padded with `<PAD>` until they reach 128 tokens.

For example:

```text
<SOS> earthquake reported <EOS> <PAD> <PAD> ... <PAD>
```

The resulting attention mask identifies real tokens and padding:

```text
Tokens:  <SOS> earthquake reported <EOS> <PAD> <PAD>
Mask:      1       1          1      1     0     0
```

### 8. Save the Tokenizer

Finally, the complete tokenizer configuration is saved:

```python
tokenizer.save("humanitarian_bpe_tokenizer.json")
```

The saved JSON contains the learned BPE vocabulary and merge rules together with the tokenizer configuration, allowing the same tokenizer to be reused during training, validation, testing, and inference.

It can later be loaded using:

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file(
    "humanitarian_bpe_tokenizer.json"
)
```

## Tokenization Pipeline

```text
Raw Tweet
   ↓
Text Cleaning
   ↓
Whitespace Pre-Tokenization
   ↓
BPE Tokenization
   ↓
<SOS> + Tokens + <EOS>
   ↓
Truncation / Padding
   ↓
Token IDs + Attention Mask
   ↓
Transformer Encoder
```