import os

import pandas as pd
import torch
import torch.nn as nn

from datasets import Dataset

NUM_WORKERS = os.cpu_count()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False


def evaluate(
    model: nn.Module,
    test_data: pd.DataFrame,
    batch_size: int,
) -> list[int]:
    test_dataset = Dataset(test_data)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
    )

    total_acc_test = 0
    results = []

    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        for test_input, test_label in test_dataloader:
            test_label = test_label.to(DEVICE)
            mask = test_input["attention_mask"].to(DEVICE)
            input_id = test_input["input_ids"].squeeze(1).to(DEVICE)

            output = model(input_id, mask)

            acc = (output.argmax(dim=1) == test_label).sum().item()
            results.extend(output.argmax(dim=1).tolist())
            total_acc_test += acc

    accuracy = round(total_acc_test / len(test_data), 3)
    print(f"Test Accuracy: {accuracy: .3f}")
    return results, accuracy
