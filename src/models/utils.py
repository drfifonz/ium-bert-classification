from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd


class utils:
    def __init__(self) -> None:
        pass

    @staticmethod
    def save_model(model: nn.Module, model_path: str) -> None:
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(model.state_dict(), model_path)
        print(f"[INFO]\t Model saved at: {model_path}")

    @staticmethod
    def load_model(model: nn.Module, model_path: str) -> nn.Module:
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        model.load_state_dict(torch.load(model_path))
        return model

    @staticmethod
    def save_results(labels: list[int], results: list[int], file_path: str) -> None:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame({"labels": labels, "results": results})
        df.to_csv(file_path, index=False)

    @staticmethod
    def save_data_to_csv(file_path: str, build_id: int, data: float) -> None:
        file_path = Path(file_path)

        df = pd.DataFrame({"build_id": [build_id], "data": [data]})
        df.to_csv(file_path, mode="a", header=False, index=False)


if __name__ == "__main__":
    x = 1.22
    build_id = 21
    utils.save_data_to_csv("./results/acc.csv", build_id, x)
