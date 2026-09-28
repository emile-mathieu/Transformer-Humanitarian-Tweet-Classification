import csv
import os

def save_metrics(
    metrics,
    filepath="results/train_metrics.csv"
):
    # Create parent folder if it doesn't exist
    parent = os.path.dirname(filepath)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(filepath, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_accuracy",
                "val_loss",
                "val_accuracy",
                "learning_rate"
            ]
        )

        writer.writeheader()
        writer.writerows(metrics)
