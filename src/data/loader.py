# src/data/loader.py

from datasets import load_dataset


class DataLoader:

    def __init__(self):
        self.dataset_name = "wikitext"
        self.dataset_version = "wikitext-2-raw-v1"

    def download(self):

        dataset = load_dataset(
            self.dataset_name,
            self.dataset_version
        )

        return dataset