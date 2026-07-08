import os
import torch
import joblib
import requests
import os
import psutil

process = psutil.Process(os.getpid())

print("=" * 50)
print("TAGOS MEMORY")
print("=" * 50)
print("Startup:",
      process.memory_info().rss / (1024 * 1024), "MB")

from transformers import DistilBertTokenizer

from bert_model import TagosModel
from config import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SAVE_DIR = os.path.join(BASE_DIR, "saved_model") + "/"

MODEL_PATH = SAVE_DIR + "tagos_model.pth"

MODEL_URL = "https://huggingface.co/Shivacer8888/tagos-model/resolve/main/tagos_model.pth"

mlb = joblib.load(SAVE_DIR + "label_encoder.pkl")
print("After label encoder:",
      process.memory_info().rss / (1024 * 1024), "MB")

from transformers import DistilBertTokenizerFast

tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
)
print("After tokenizer:",
      process.memory_info().rss / (1024 * 1024), "MB")

model = TagosModel(
    MODEL_NAME,
    len(mlb.classes_)
)
print("After model creation:",
      process.memory_info().rss / (1024 * 1024), "MB")


# Download model only if missing
if not os.path.exists(MODEL_PATH):

    print("Downloading TAGOS model...")

    response = requests.get(MODEL_URL)

    response.raise_for_status()

    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)

    print("Download completed.")

# ================= DEBUG =================

import hashlib

print("=" * 60)
print("TAGOS DEBUG")
print("=" * 60)

print("MODEL PATH:", MODEL_PATH)
print("FILE EXISTS:", os.path.exists(MODEL_PATH))

if os.path.exists(MODEL_PATH):

    print("FILE SIZE:", os.path.getsize(MODEL_PATH))

    with open(MODEL_PATH, "rb") as f:
        first100 = f.read(100)

    print("FIRST 100 BYTES:", first100)

    with open(MODEL_PATH, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()

    print("SHA256:", sha)

state_dict = torch.load(
    MODEL_PATH,
    map_location="cpu"
)
print("After torch.load:",
      process.memory_info().rss / (1024 * 1024), "MB")

print("MODEL LOADED SUCCESSFULLY")

model.load_state_dict(state_dict)
print("After load_state_dict:",
      process.memory_info().rss / (1024 * 1024), "MB")

print("Memory after loading model:",
      process.memory_info().rss / (1024 * 1024), "MB")

model.eval()

print("FINAL:",
      process.memory_info().rss / (1024 * 1024), "MB")

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