import torch
import torch.nn as nn

from models.model import TweetTransformer
from data.dataloader import create_train_dataloader, create_eval_dataloader


def train_model(
    model,
    train_data,
    val_data,
    tokenizer_path,
    batch_size=32,
    num_epochs=15,
    d_model=512,
    warmup_steps=4000
):
    # Create DataLoaders
    train_loader = create_train_dataloader(
        train_data,
        tokenizer_path,
        batch_size=batch_size
    )

    eval_loader = create_eval_dataloader(
        val_data,
        tokenizer_path,
        batch_size=batch_size
    )

    # Multi-class classification loss
    criterion = nn.CrossEntropyLoss()

    # Adam configuration from Attention Is All You Need
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1.0,
        betas=(0.9, 0.98),
        eps=1e-9
    )

    # Transformer learning-rate schedule
    def transformer_lr(step):
        step = max(step, 1)

        return (d_model ** -0.5) * min(
            step ** -0.5,
            step * (warmup_steps ** -1.5)
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=transformer_lr
    )

    # Early stopping
    best_val_loss = float("inf")
    patience = 3
    patience_counter = 0

    # Store metrics for every epoch
    metrics = []

    for epoch in range(num_epochs):

        # ==========================
        # Training
        # ==========================
        model.train()

        train_total_loss = 0.0
        train_total_correct = 0
        train_total_samples = 0

        for batch in train_loader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            optimizer.zero_grad()

            # Shape: [batch_size, num_classes]
            outputs = model(input_ids, attention_mask)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()
            scheduler.step()

            batch_size_actual = labels.size(0)

            # CrossEntropyLoss is the mean loss for this batch.
            # Multiply by batch size to recover the summed loss
            # across samples in this batch.
            train_total_loss += (
                loss.item() * batch_size_actual
            )

            predictions = outputs.argmax(dim=1)

            train_total_correct += (
                predictions == labels
            ).sum().item()

            train_total_samples += batch_size_actual

        # Sample-weighted epoch metrics
        train_avg_loss = (
            train_total_loss / train_total_samples
        )

        train_avg_accuracy = (
            train_total_correct / train_total_samples
        )

        # ==========================
        # Validation
        # ==========================
        model.eval()

        total_val_loss = 0.0
        total_val_correct = 0
        total_val_samples = 0

        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]

                outputs = model(
                    input_ids,
                    attention_mask
                )

                val_loss = criterion(
                    outputs,
                    labels
                )

                batch_size_actual = labels.size(0)

                total_val_loss += (
                    val_loss.item() * batch_size_actual
                )

                predictions = outputs.argmax(dim=1)

                total_val_correct += (
                    predictions == labels
                ).sum().item()

                total_val_samples += batch_size_actual

        val_avg_loss = (
            total_val_loss / total_val_samples
        )

        val_avg_accuracy = (
            total_val_correct / total_val_samples
        )

        # ==========================
        # Metrics
        # ==========================
        current_lr = scheduler.get_last_lr()[0]

        metrics.append({
            "epoch": epoch + 1,
            "train_loss": train_avg_loss,
            "train_accuracy": train_avg_accuracy,
            "val_loss": val_avg_loss,
            "val_accuracy": val_avg_accuracy,
            "learning_rate": current_lr
        })

        # ==========================
        # Early stopping
        # ==========================
        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            patience_counter = 0

            torch.save(
                model.state_dict(),
                "best_model.pth"
            )

        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(
                f"Early stopping at epoch {epoch + 1}: "
                f"validation loss did not improve for "
                f"{patience} consecutive epochs."
            )
            break

    return metrics