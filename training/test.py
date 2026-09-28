import torch
import torch.nn as nn

from sklearn.metrics import recall_score, f1_score

from data.dataloader import create_eval_dataloader, create_test_dataloader

def evaluate_model(
    device,
    model,
    val_data,
    test_data,
    tokenizer_path,
    batch_size=32
):
    val_loader = create_eval_dataloader(
        val_data,
        tokenizer_path,
        batch_size=batch_size
    )

    test_loader = create_test_dataloader(
        test_data,
        tokenizer_path,
        batch_size=batch_size
    )

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

        # ---------------- VALIDATION ----------------
        for batch in val_loader:
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)

            outputs = model(
                input_ids,
                attention_mask
            )

            loss = criterion(outputs, labels)

            batch_size_actual = labels.size(0)

            # loss.item() is the mean loss for this batch,
            # so multiply by the number of samples in the batch
            val_total_loss += loss.item() * batch_size_actual

            predicted = torch.argmax(
                outputs,
                dim=1
            )

            val_total_correct += (
                predicted == labels
            ).sum().item()

            val_total_samples += batch_size_actual

            val_predictions.extend(
                predicted.cpu().tolist()
            )

            val_targets.extend(
                labels.cpu().tolist()
            )

        # ---------------- TEST ----------------
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids,
                attention_mask
            )

            loss = criterion(outputs, labels)

            batch_size_actual = labels.size(0)

            test_total_loss += (
                loss.item() * batch_size_actual
            )

            predicted = torch.argmax(
                outputs,
                dim=1
            )

            test_total_correct += (
                predicted == labels
            ).sum().item()

            test_total_samples += batch_size_actual

            test_predictions.extend(
                predicted.cpu().tolist()
            )

            test_targets.extend(
                labels.cpu().tolist()
            )

    # Sample-weighted average loss
    val_avg_loss = (
        val_total_loss / val_total_samples
    )

    val_accuracy = (
        val_total_correct / val_total_samples
    )

    val_recall = recall_score(
        val_targets,
        val_predictions,
        average="macro",
        zero_division=0
    )

    val_f1 = f1_score(
        val_targets,
        val_predictions,
        average="macro",
        zero_division=0
    )

    test_avg_loss = (
        test_total_loss / test_total_samples
    )

    test_accuracy = (
        test_total_correct / test_total_samples
    )

    test_recall = recall_score(
        test_targets,
        test_predictions,
        average="macro",
        zero_division=0
    )

    test_f1 = f1_score(
        test_targets,
        test_predictions,
        average="macro",
        zero_division=0
    )

    metrics = {
        "val_loss": val_avg_loss,
        "val_accuracy": val_accuracy,
        "val_recall": val_recall,
        "val_f1_score": val_f1,
        "test_loss": test_avg_loss,
        "test_accuracy": test_accuracy,
        "test_recall": test_recall,
        "test_f1_score": test_f1,
    }

    return metrics