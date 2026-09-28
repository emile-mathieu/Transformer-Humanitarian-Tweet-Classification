from torch.utils.data import DataLoader
from data.dataset import CustomDataset

def create_train_dataloader(data, tokenizer_path, batch_size=32, shuffle=True):
    train_dataset = CustomDataset(data, tokenizer_path)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

def create_eval_dataloader(data, tokenizer_path, batch_size=32):
    eval_dataset = CustomDataset(data, tokenizer_path)
    return DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

def create_test_dataloader(data, tokenizer_path, batch_size=32):
    test_dataset = CustomDataset(data, tokenizer_path)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  