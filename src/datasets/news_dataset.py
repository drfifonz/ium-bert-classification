import pandas as pd
from typing import Callable
from pathlib import Path


class NewsDataset:
    def __init__(self, data_dir_path: str = "data", data_lenght: int = None) -> None:
        self.data_dir_path = Path("./" + data_dir_path)
        self.dataset_dir_path = self.data_dir_path / "dataset"
        self.true_news_path = self.data_dir_path / "True.csv"
        self.fake_news_path = self.data_dir_path / "Fake.csv"

        self.__data_lenght = data_lenght

        self.true_news = self.__load_news(self.true_news_path, self.__data_lenght)
        self.fake_news = self.__load_news(self.fake_news_path, self.__data_lenght)

        self.true_news["label"] = 1
        self.fake_news["label"] = 0

        self.train, self.test, self.val = self.__load_train_val_test()

    def __load_news(self, file_path: Path, trim: int = None) -> pd.DataFrame:
        news = pd.read_csv(file_path)
        news = news.drop(columns=["title", "subject", "date"])

        return news if not trim else news.head(trim)

    def __load_data(self, file_path: Path) -> pd.DataFrame:
        df = pd.read_csv(file_path).rename(columns={"Value": "label"})
        return self.__convert_text(df)

    def __load_train_val_test(self):
        train_data = self.__load_data(self.dataset_dir_path / "train.csv")
        test_data = self.__load_data(self.dataset_dir_path / "test.csv")
        val_data = self.__load_data(self.dataset_dir_path / "val.csv")

        total_data_len = train_data.size + test_data.size + val_data.size

        trim: Callable[[pd.DataFrame], pd.DataFrame] = lambda df: df.head(
            int(self.__data_lenght * 2 * df.size / total_data_len)
        )
        train_data = trim(train_data)
        test_data = trim(test_data)
        val_data = trim(val_data)
        return train_data, test_data, val_data

    def __convert_text(self, df: pd.DataFrame) -> pd.DataFrame:
        df["text"] = df["text"].str.strip().astype(str)
        df.dropna(axis=0, how="any", inplace=False, subset=["text"])
        return df

    @property
    def data(self) -> pd.DataFrame:
        dataset = pd.concat([self.true_news, self.fake_news], axis=0)
        return self.__convert_text(dataset)


if __name__ == "__main__":
    dataset = NewsDataset(data_lenght=1000)
    print("dataset")
    print(dataset.data.size)
    print("train")
    print(dataset.train.size)
    print("val")
    print(dataset.test.size)
    print("test")
    print(dataset.val.size)
