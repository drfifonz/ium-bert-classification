import argparse


import torch

from datasets import NewsDataset
from evaluate import evaluate
from models import BertClassifier, utils
from train import train

# argument parser
parser = argparse.ArgumentParser(
    prog="News classification",
    description="Train or evaluate model",
)
parser.add_argument("--train", action="store_true", default=False)
parser.add_argument("--test", action="store_true", default=False)
parser.add_argument("--model_path", type=str, default="results/model.pt")
parser.add_argument("--results_path", type=str, default="results/results.csv")
parser.add_argument("--data_acc_path", type=str, default="./results/acc.csv")
parser.add_argument("--build_id", type=str, default="0")


# HYPER PARAMETERS
parser.add_argument("--batch", "-b", type=int, default=2)
parser.add_argument("--learning_rate", "--lr", type=float, default=1e-6)
parser.add_argument("--num_epochs", "--epochs", "-e", type=int, default=3)
parser.add_argument("--data_len", type=int, default=1000)


def main():
    args = parser.parse_args()

    INITIAL_LR = args.learning_rate
    NUM_EPOCHS = args.num_epochs
    BATCH_SIZE = args.batch
    DATA_LEN = args.data_len
    print("INITIAL_LR: ", INITIAL_LR)
    print("NUM_EPOCHS: ", NUM_EPOCHS)
    print("BATCH_SIZE: ", BATCH_SIZE)
    print("DATA_LEN: ", DATA_LEN)
    print("CUDA: ", torch.cuda.is_available())

    # raise
    # loading & spliting data
    news_dataset = NewsDataset(data_dir_path="data", data_lenght=DATA_LEN)

    train_data = news_dataset.train
    test_data = news_dataset.test
    val_data = news_dataset.val

    # train_val_data, test_data = train_test_split(
    #     news_dataset.data,
    #     test_size=0.2,
    #     shuffle=True,
    #     random_state=random.seed(SEED),
    # )

    # train_data, val_data = train_test_split(
    #     train_val_data,
    #     test_size=0.2,
    #     shuffle=True,
    #     random_state=random.seed(SEED),
    # )

    # trainig model
    if args.train:
        trained_model, metrics = train(
            model=BertClassifier(),
            train_data=train_data,
            val_data=val_data,
            learning_rate=INITIAL_LR,
            epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
        )
        utils.save_model(model=trained_model, model_path=args.model_path)

    # evaluating model
    if args.test:
        model = utils.load_model(model=BertClassifier(), model_path=args.model_path)  # loading model from model.pt file
        results, accuracy = evaluate(
            model=model,
            test_data=test_data,
            batch_size=BATCH_SIZE,
        )
        utils.save_results(labels=test_data["label"], results=results, file_path=args.results_path)
        utils.save_data_to_csv(
            file_path=args.data_acc_path,
            build_id=int(args.build_id),
            data=accuracy,
        )


if __name__ == "__main__":
    main()
