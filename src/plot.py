import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator


def draw_and_save_plot(build_ids: list[int], acc_values: list[float], file_path: str):
    file_path = Path(file_path)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(build_ids, [acc * 100 for acc in acc_values])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Build Id")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy for builds")
    fig.savefig(file_path)
    plt.close(fig)


parser = argparse.ArgumentParser(
    prog="plot parser",
    description="Setting parameter for plot",
)
parser.add_argument("--data_acc_path", type=str, default="./results/acc.csv")
parser.add_argument("--plot_acc_path", type=str, default="./results/plot.png")

if __name__ == "__main__":
    args = parser.parse_args()
    ACC_FILE_PATH = Path(args.data_acc_path)
    ACC_PLOT_PATH = Path(args.plot_acc_path)

    df = pd.read_csv(ACC_FILE_PATH, header=None)
    draw_and_save_plot(
        build_ids=df.iloc[:, 0],
        acc_values=df.iloc[:, 1],
        file_path=ACC_PLOT_PATH,
    )
