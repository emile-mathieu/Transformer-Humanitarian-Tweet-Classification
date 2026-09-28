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
        "test_f1_score"
    ]

    df = pd.DataFrame([metrics])

    missing_columns = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing evaluation metric columns: {missing_columns}"
        )

    df = df[columns]

    _ensure_parent_directory(filepath)

    df.to_csv(filepath, index=False)
