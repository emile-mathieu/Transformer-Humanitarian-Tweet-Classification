import torch
import torch.nn as nn

from models.model import TweetTransformer
from data.dataloaders import create_test_dataloader

from utils import save_metrics

def test_model(model, test_data, tokenizer_path, batch_size=32):
    # Create the DataLoader for test data
    test_loader = create_test_dataloader(test_data, tokenizer_path, batch_size=batch_size)

    # Cross entropy for multi-class classification
    criterion = nn.CrossEntropyLoss()

    model.eval()

    test_total_loss = 0.0
    test_total_correct = 0
    test_total_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            # Output size is (batch_size, num_classes)
            outputs = model(input_ids, attention_mask)

            loss = criterion(outputs, labels)
            test_total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            test_total_correct += (predicted == labels).sum().item()
            test_total_samples += labels.size(0)

    avg_test_loss = test_total_loss / len(test_loader)
    test_accuracy = test_total_correct / test_total_samples

    metrics = {
        "test_loss": avg_test_loss,
        "test_accuracy": test_accuracy
    }

    save_metrics(metrics)