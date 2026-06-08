import os
import pickle
import torch

from src.models.bilstm_attention import BiLSTMAttention
from src.utils.download_model import download_model


class BiLSTMPredictor:

    def __init__(self):

        ROOT_DIR = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                ".."
            )
        )

        VOCAB_PATH = os.path.join(
            ROOT_DIR,
            "artifacts",
            "vocab",
            "vocab.pkl"
        )

        if not os.path.exists(VOCAB_PATH):
            raise FileNotFoundError(
                f"Vocabulary file not found: {VOCAB_PATH}"
            )

        with open(VOCAB_PATH, "rb") as f:
            vocab = pickle.load(f)

        self.word_to_idx = vocab.word_to_idx
        self.idx_to_word = vocab.idx_to_word

        print("Downloading model from Hugging Face...")

        model_path = download_model()

        print("Model downloaded successfully")

        self.model = BiLSTMAttention(
            vocab_size=len(self.word_to_idx)
        )

        self.model.load_state_dict(
            torch.load(
                model_path,
                map_location="cpu"
            )
        )

        self.model.eval()

    def predict_top_k(
        self,
        text,
        k=5,
        sequence_length=10
    ):

        words = text.lower().split()

        tokens = [
            self.word_to_idx.get(
                word,
                self.word_to_idx["<UNK>"]
            )
            for word in words
        ]

        if len(tokens) < sequence_length:

            tokens = (
                [0] * (sequence_length - len(tokens))
                + tokens
            )

        else:

            tokens = tokens[-sequence_length:]

        x = torch.tensor(
            [tokens],
            dtype=torch.long
        )

        with torch.no_grad():

            output = self.model(x)

            probs = torch.softmax(
                output,
                dim=1
            )

            top_probs, top_idx = torch.topk(
                probs,
                k
            )

        predictions = []

        for prob, idx in zip(
            top_probs[0],
            top_idx[0]
        ):

            predictions.append(
                (
                    self.idx_to_word[idx.item()],
                    prob.item()
                )
            )

        return predictions