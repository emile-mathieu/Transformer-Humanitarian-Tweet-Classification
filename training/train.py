import torch
import torch.nn as nn

from models.model import TweetTransformer
from data.dataloader import create_train_dataloader, create_eval_dataloader

from utils import save_metrics


def train_model(model,train_data, tokenizer_path, batch_size=32, num_epochs=15, d_model=512, warmup_steps=4000):
    # Create the DataLoader for training data
    train_loader = create_train_dataloader(train_data, tokenizer_path, batch_size=batch_size)

    # Create the DataLoader for evaluation data
    eval_loader = create_eval_dataloader(train_data, tokenizer_path, batch_size=batch_size)

    # Cross entropy for multi-class classification
    criterion = nn.CrossEntropyLoss()

    # Adam configuration used in Attention Is All You Need
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1.0,
        betas=(0.9, 0.98),
        eps=1e-9
    )

    # Learning-rate schedule from Attention Is All You Need
    def transformer_lr(step):
        step = max(step, 1)

        return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=transformer_lr
    )
    # Early stopping parameters
    best_val_loss = float("inf")
    patience = 3
    patience_counter = 0

    # Store metrics for every epoch
    metrics = []

    for epoch in range(num_epochs):
        model.train()

        train_total_loss = 0.0
        train_total_correct = 0
        train_total_samples = 0

        for batch in train_loader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            optimizer.zero_grad()

            # Output size is (batch_size, num_classes)
            outputs = model(input_ids, attention_mask)

            loss = criterion(outputs, labels)
            loss.backward()

            # Update parameters
            optimizer.step()

            # Update learning rate
            scheduler.step()

            # .item() is used to get the Python number from a tensor with a single value (not float). 
            # Track loss
            train_total_loss += loss.item()

            # Track accuracy
            # dim=1 because we have num of dimensions = 2, (batch_size, num_classes), and we want to get the index of the max value along the class dimension
            predictions = outputs.argmax(dim=1)

            # Same here, need to convert to item() to get the number of correct predictions
            train_total_correct += (predictions == labels).sum().item()

            # label.size(0) gives the number of samples in the batch, which is used to calculate accuracy
            train_total_samples += labels.size(0)

        train_avg_loss = train_total_loss / len(train_loader)
        train_avg_accuracy = train_total_correct / train_total_samples

        # Validation step after each epoch
        model.eval()

        total_val_loss = 0.0
        total_val_correct = 0
        total_val_samples = 0

        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]

                outputs = model(input_ids, attention_mask)

                val_loss = criterion(outputs, labels)
                total_val_loss += val_loss.item()

                predictions = outputs.argmax(dim=1)
                total_val_correct += (predictions == labels).sum().item()
                total_val_samples += labels.size(0)

        val_avg_loss = total_val_loss / len(eval_loader)
        val_avg_accuracy = total_val_correct / total_val_samples

        # Metrics for the current epoch
        current_lr = scheduler.get_last_lr()[0]
        metrics.append({
            "epoch": epoch + 1,
            "train_loss": train_avg_loss,
            "train_accuracy": train_avg_accuracy,
            "val_loss": val_avg_loss,
            "val_accuracy": val_avg_accuracy,
            "learning_rate": current_lr
        })

        # Early Stopping Setup:
        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            patience_counter = 0

            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1} due to low validation loss.")
            break

    save_metrics(metrics)