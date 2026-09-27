# testing dataset loading
from dataset import CustomDataset


if __name__ == "__main__":
    # Example usage
    data = [
        {"tweet_text": "Check out this link: http://example.com", "class_label": 1},
        {"tweet_text": "Hello @user, how are you?", "class_label": 0},
        {"tweet_text": "This is a test tweet with &amp;", "class_label": 2},
    ]
    # Change tokenizer path if necessary.
    dataset = CustomDataset(data, tokenizer_path="../tokenizer/humanitarian_bpe_tokenizer.json")

    for i in range(len(dataset)):
        item = dataset[i]
        decoded = dataset.tokenizer.decode(
        item["input_ids"].tolist(),
        skip_special_tokens=True
        )
        print(f"Decoded text: {decoded}")
        print(f"Input IDs: {item['input_ids']}")
        print(f"Attention Mask: {item['attention_mask']}")
        print(f"Labels: {item['labels']}")