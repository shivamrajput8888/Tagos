import os
import torch
import joblib
import requests

from transformers import DistilBertTokenizer

from bert_model import TagosModel
from config import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SAVE_DIR = os.path.join(BASE_DIR, "saved_model") + "/"

MODEL_PATH = SAVE_DIR + "tagos_model.pth"

MODEL_URL = "https://huggingface.co/Shivacer8888/tagos-model/resolve/main/tagos_model.pth"

mlb = joblib.load(SAVE_DIR + "label_encoder.pkl")

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")


if not os.path.exists(MODEL_PATH):

    print("Downloading TAGOS model...")

    response = requests.get(MODEL_URL, stream=True)

    response.raise_for_status()

    with open(MODEL_PATH, "wb") as f:

        for chunk in response.iter_content(chunk_size=8192):

            if chunk:
                f.write(chunk)

    print("Model downloaded successfully!")

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu"
    )
)

model.eval()


def predict_tags(text, top_k=5):

    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model(
            encoding["input_ids"],
            encoding["attention_mask"]
        )

        probabilities = torch.sigmoid(outputs).numpy()[0]

    indices = probabilities.argsort()[-top_k:][::-1]

    top_probs = [float(probabilities[i]) for i in indices]

    min_prob = min(top_probs)
    max_prob = max(top_probs)

    predictions = []

    for idx in indices:

        prob = float(probabilities[idx])

        if max_prob == min_prob:
            display_score = 95.0
        else:
            display_score = 85 + ((prob - min_prob) / (max_prob - min_prob)) * 14

        predictions.append(
            (
                mlb.classes_[idx],
                round(display_score, 2)
            )
        )

    return predictions