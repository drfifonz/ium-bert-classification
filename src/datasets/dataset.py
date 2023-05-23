import torch
import pandas as pd
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data: pd.DataFrame) -> None:
        self.labels = data["label"].to_list()
        self.texts = data["text"].to_list()

    def __getitem__(self, idx):
        label = self.labels[idx]
        text = tokenizer(
            self.texts[idx],
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )

        return text, label

    def __len__(self) -> int:
        return len(self.labels)
