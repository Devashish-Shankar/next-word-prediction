import sys
import os

ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

sys.path.append(ROOT_DIR)

from src.inference.bilstm_predict import BiLSTMPredictor

predictor = BiLSTMPredictor()


def get_predictions(text):

    return predictor.predict_top_k(
        text=text,
        k=5
    )