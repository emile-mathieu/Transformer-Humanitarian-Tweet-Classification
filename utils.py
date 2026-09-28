import os
import pandas as pd


def save_training_history(
    metrics,
    filepath="results/train_metrics.csv"
):
    """
    Save per-epoch training history.

    Expected:
    [
        {
            "epoch": 1,
            "train_loss": ...,
            "train_accuracy": ...,
            "val_loss": ...,
            "val_accuracy": ...,
            "learning_rate": ...
        },
        ...
    ]
    """

    columns = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "learning_rate",
    ]

    if isinstance(metrics, dict):
        metrics = [metrics]

    df = pd.DataFrame(metrics)

    missing_columns = [
        column for column in columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing training metric columns: {missing_columns}"
        )

    df = df[columns]

    parent = os.path.dirname(filepath)
    if parent:
        os.makedirs(parent, exist_ok=True)

    df.to_csv(filepath, index=False)


def save_evaluation_metrics(
    metrics,
    filepath="results/evaluation_metrics.csv"
):
    """
    Save final validation and test metrics.
    """

    columns = [
        "val_loss",
        "val_accuracy",
        "val_recall",
        "val_f1_score",
        "test_loss",
        "test_accuracy",
        "test_recall",
        "test_f1_score",
    ]

    df = pd.DataFrame([metrics])

    missing_columns = [
        column for column in columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing evaluation metric columns: {missing_columns}"
        )

    df = df[columns]

    parent = os.path.dirname(filepath)
    if parent:
        os.makedirs(parent, exist_ok=True)

    df.to_csv(filepath, index=False)