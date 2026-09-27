import csv

def save_metrics(
    metrics,
    filepath="results/train_metrics.csv"
):
    with open(filepath, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "loss",
                "accuracy",
                "learning_rate"
            ]
        )

        writer.writeheader()
        writer.writerows(metrics)
