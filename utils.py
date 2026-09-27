import csv
import os

def save_metrics(
    metrics,
    filepath="results/train_metrics.csv"
):
    # Create parent folder if it doesn't exist
    os.makedirs(
        os.path.dirname(filepath),
        exist_ok=True
    )

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
