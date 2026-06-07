# src/data/ingestion.py

import os

from loader import DataLoader
from validator import DataValidator


class DataIngestion:

    def __init__(self):

        self.output_path = "data/raw/wikitext.txt"

    def run(self):

        loader = DataLoader()

        dataset = loader.download()

        DataValidator.validate(dataset)

        train_text = "\n".join(
            dataset["train"]["text"]
        )

        os.makedirs(
            os.path.dirname(self.output_path),
            exist_ok=True
        )

        with open(
            self.output_path,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(train_text)

        print(
            f"Dataset saved at {self.output_path}"
        )


if __name__ == "__main__":

    ingestion = DataIngestion()

    ingestion.run()