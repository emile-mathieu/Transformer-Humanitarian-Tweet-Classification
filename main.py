import os
import torch
import torch.nn as nn

from datasets import load_dataset

from models.model import TweetTransformer

from data.dataloader import create_train_dataloader, create_eval_dataloader

from training.train import train_model
from training.test import evaluate_model

from utils import save_training_history, save_evaluation_metrics


def load_data():
    data_files = {
    "train": "https://huggingface.co/datasets/QCRI/HumAID-all/resolve/main/data/train-00000-of-00001.parquet",
    "validation": "https://huggingface.co/datasets/QCRI/HumAID-all/resolve/main/data/validation-00000-of-00001.parquet",
    "test": "https://huggingface.co/datasets/QCRI/HumAID-all/resolve/main/data/test-00000-of-00001.parquet",
}
    ds = load_dataset("parquet", data_files=data_files)
    return ds

def main() -> None:
    # Load the dataset
    dataset = load_data()
    
    train_data = dataset["train"]
    val_data = dataset["validation"]
    test_data = dataset["test"]
    
    current_path = os.path.dirname(os.path.abspath(__file__))
    tokenizer_path = os.path.join(current_path, "tokenizer", "tokenizer.json")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TweetTransformer().to(device)

    # Train the model
    training_metrics = train_model(device, model, train_data, val_data, tokenizer_path)

    # Load the best model for evaluation
    model.load_state_dict(torch.load("best_model.pth"))
    
    # Evaluate the model
    evaluation_metrics = evaluate_model(device, model, val_data, test_data, tokenizer_path)
    
    save_training_history(training_metrics)
    save_evaluation_metrics(evaluation_metrics)

if __name__ == "__main__":
	main()
