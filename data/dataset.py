import re
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer
from pathlib import Path

def clean_text(text):
    # Normalize whitespace
    text = ' '.join(text.split())
    # Replace URLs with <URL>
    text = re.sub(r'http\S+|www\S+|https\S+', '<URL>', text, flags=re.MULTILINE)
    # Replace mentions with <MENTION>
    text = re.sub(r'@\w+', '<MENTION>', text)
    # Replace &amp; with &
    text = text.replace('&amp;', '&')

    return text

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer_path):
        self.data = data
        self.max_length = 128 

        # Check path to tokenizer json
        tokenizer_path = Path(tokenizer_path)
        if not tokenizer_path.is_file():
            raise FileNotFoundError(
                f"Tokenizer file was not found: {tokenizer_path}. "
                "Run tokenizer/tokenizer.ipynb and save the tokenizer first."
            )
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        
        # Enable padding with the specified length because trained tokenizer has dynamic padding by default.
        self.tokenizer.enable_padding(
            # Important to specify length here, or else dynamic padding used.
            length=self.max_length,
            pad_id=self.tokenizer.token_to_id("<PAD>"),
            pad_token="<PAD>",
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]

        try:
            text = clean_text(row["tweet_text"])
            label = row["class_label"]
        except KeyError as error:
            raise KeyError(
                "Each row must contain 'tweet_text' and 'class_label'."
            ) from error

        encoding = self.tokenizer.encode(text)
        # Converting HF arryys to torch tensors
        return {
            "input_ids": torch.tensor(encoding.ids, dtype=torch.long),
            "attention_mask": torch.tensor(
                encoding.attention_mask, dtype=torch.long
            ),
            "labels": torch.tensor(label, dtype=torch.long),
        }

