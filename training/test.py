import torch
import torch.nn as nn

from data.dataloader import create_test_dataloader

from sklearn.metrics import recall_score, f1_score


def test_model(model, val_data, test_data, tokenizer_path, batch_size=32):
    val_loader = create_test_dataloader(val_data, tokenizer_path, batch_size=batch_size)
    test_loader = create_test_dataloader(test_data, tokenizer_path, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    model.eval()

    val_total_loss = 0.0
    val_total_correct = 0
    val_total_samples = 0
    val_predictions = []
    val_targets = []

    test_total_loss = 0.0
    test_total_correct = 0
    test_total_samples = 0
    test_predictions = []
    test_targets = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            val_total_loss += loss.item()

            predicted = torch.argmax(outputs, dim=1)

            val_total_correct += (predicted == labels).sum().item()
            val_total_samples += labels.size(0)

            val_predictions.extend(predicted.cpu().tolist())
            val_targets.extend(labels.cpu().tolist())

        for batch in test_loader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            test_total_loss += loss.item()

            predicted = torch.argmax(outputs, dim=1)

            test_total_correct += (predicted == labels).sum().item()
            test_total_samples += labels.size(0)

            test_predictions.extend(predicted.cpu().tolist())
            test_targets.extend(labels.cpu().tolist())

    val_avg_loss = val_total_loss / len(val_loader)
    val_accuracy = val_total_correct / val_total_samples
    val_recall = recall_score(val_targets, val_predictions, average="macro", zero_division=0)
    val_f1 = f1_score(val_targets, val_predictions, average="macro", zero_division=0)

    test_avg_loss = test_total_loss / len(test_loader)
    test_accuracy = test_total_correct / test_total_samples
    test_recall = recall_score(test_targets, test_predictions, average="macro", zero_division=0)
    test_f1 = f1_score(test_targets, test_predictions, average="macro", zero_division=0)

    metrics = {
        "val_loss": val_avg_loss,
        "val_accuracy": val_accuracy,
        "val_recall": val_recall,
        "val_f1_score": val_f1,
        "test_loss": test_avg_loss,
        "test_accuracy": test_accuracy,
        "test_recall": test_recall,
        "test_f1_score": test_f1
    }

    return metrics