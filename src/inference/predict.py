import torch

from src.models.lstm import LSTMNextWordPredictor


class Predictor:

    def __init__(self, checkpoint_path):

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu"
        )

        self.word_to_idx = checkpoint["word_to_idx"]
        self.idx_to_word = checkpoint["idx_to_word"]

        self.model = LSTMNextWordPredictor(
            vocab_size=checkpoint["vocab_size"],
            embedding_dim=checkpoint["embedding_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            num_layers=checkpoint["num_layers"]
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.eval()

    def predict_next_word(self, text):

        words = text.lower().split()

        sequence = [
            self.word_to_idx.get(
                word,
                self.word_to_idx["<UNK>"]
            )
            for word in words
        ]

        if len(sequence) < 5:

            sequence = (
                [0] * (5 - len(sequence))
                + sequence
            )

        else:

            sequence = sequence[-5:]

        x = torch.tensor(
            [sequence],
            dtype=torch.long
        )

        with torch.no_grad():

            output = self.model(x)

            prediction = torch.argmax(
                output,
                dim=1
            ).item()

        return self.idx_to_word[prediction]