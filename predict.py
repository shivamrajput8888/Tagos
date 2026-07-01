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

tokenizer = DistilBertTokenizer.from_pretrained(
    "distilbert-base-uncased"
)

model = TagosModel(
    MODEL_NAME,
    len(mlb.classes_)
)

# Download model only if missing
if not os.path.exists(MODEL_PATH):

    print("Downloading TAGOS model...")

    response = requests.get(MODEL_URL)

    response.raise_for_status()

    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)

    print("Download completed.")

# ================= DEBUG =================

print("=" * 50)
print("MODEL PATH:", MODEL_PATH)
print("EXISTS:", os.path.exists(MODEL_PATH))

if os.path.exists(MODEL_PATH):
    print("SIZE:", os.path.getsize(MODEL_PATH))

    with open(MODEL_PATH, "rb") as f:
        first = f.read(100)

    print("FIRST BYTES:", first)

# ======================================== 

state_dict = torch.load(
    MODEL_PATH,
    map_location="cpu"
)

model.load_state_dict(state_dict)

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