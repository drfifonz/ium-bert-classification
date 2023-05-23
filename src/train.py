import os

import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from datasets import Dataset

NUM_WORKERS = os.cpu_count()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False


def train(
    model: nn.Module,
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    learning_rate: float,
    epochs: int,
    batch_size: int,
) -> nn.Module:
    train_dataset, val_dataset = Dataset(train_data), Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.to(DEVICE)
    for epoch in range(epochs):
        total_acc_train = 0
        total_loss_train = 0
        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(DEVICE)
            mask = train_input["attention_mask"].to(DEVICE)
            input_id = train_input["input_ids"].squeeze(1).to(DEVICE)

            output = model(input_id, mask)

            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():
            for val_input, val_label in val_dataloader:
                val_label = val_label.to(DEVICE)
                mask = val_input["attention_mask"].to(DEVICE)
                input_id = val_input["input_ids"].squeeze(1).to(DEVICE)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        print(
            f"Epochs: {epoch + 1}/{epochs} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                | Val Loss: {total_loss_val / len(val_data): .3f} \
                | Val Accuracy: {total_acc_val / len(val_data): .3f}"
        )
        metrics = {
            "train_acc": total_acc_train / len(train_data),
            "train_loss": total_loss_train / len(train_data),
            "val_acc": total_acc_val / len(val_data),
            "val_loss": total_loss_val / len(val_data),
        }
    return model, metrics
