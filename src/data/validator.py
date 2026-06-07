# src/data/validator.py

class DataValidator:

    @staticmethod
    def validate(dataset):

        if dataset is None:
            raise ValueError("Dataset not loaded")

        if len(dataset["train"]) == 0:
            raise ValueError("Training dataset empty")

        print("Dataset Validation Passed")

        return True