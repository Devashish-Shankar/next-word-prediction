import os
from huggingface_hub import hf_hub_download


def download_model():

    model_path = hf_hub_download(
        repo_id="devashishshankar/next-word-prediction-model",
        filename="bilstm_attention_model.pth",
        resume_download=True
    )

    return model_path