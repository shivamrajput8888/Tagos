import os
import joblib
import requests
import numpy as np
import onnxruntime as ort

from transformers import DistilBertTokenizerFast
from config import MAX_LENGTH


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "saved_model")

MODEL_PATH = os.path.join(
    SAVE_DIR,
    "tagos_model_int8.onnx"
)

LABEL_PATH = os.path.join(
    SAVE_DIR,
    "label_encoder.pkl"
)

MODEL_URL = (
    "https://huggingface.co/"
    "Shivacer8888/tagos-model/resolve/main/"
    "tagos_model_int8.onnx"
)


# Create saved_model folder if needed
os.makedirs(SAVE_DIR, exist_ok=True)


# Download ONNX model only if missing
if not os.path.exists(MODEL_PATH):
    print("Downloading INT8 ONNX model...")

    with requests.get(
        MODEL_URL,
        stream=True,
        timeout=300
    ) as response:

        response.raise_for_status()

        with open(MODEL_PATH, "wb") as file:
            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):
                if chunk:
                    file.write(chunk)

    print("ONNX model downloaded successfully!")


# Load label encoder
print("Loading label encoder...")
mlb = joblib.load(LABEL_PATH)


# Load tokenizer
print("Loading tokenizer...")
tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
)


# Load ONNX Runtime session
print("Loading ONNX model...")

session_options = ort.SessionOptions()
session_options.intra_op_num_threads = 1
session_options.inter_op_num_threads = 1

session = ort.InferenceSession(
    MODEL_PATH,
    sess_options=session_options,
    providers=["CPUExecutionProvider"]
)

print("ONNX model loaded successfully!")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def predict_tags(text, top_k=5):

    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="np"
    )

    input_ids = encoding["input_ids"].astype(np.int64)
    attention_mask = encoding["attention_mask"].astype(np.int64)

    outputs = session.run(
        None,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
    )

    logits = outputs[0][0]
    probabilities = sigmoid(logits)

    indices = probabilities.argsort()[-top_k:][::-1]

    top_probs = [
        float(probabilities[i])
        for i in indices
    ]

    min_prob = min(top_probs)
    max_prob = max(top_probs)

    predictions = []

    for idx in indices:
        prob = float(probabilities[idx])

        if max_prob == min_prob:
            display_score = 95.0
        else:
            display_score = (
                85
                + (
                    (prob - min_prob)
                    / (max_prob - min_prob)
                ) * 14
            )

        predictions.append(
            (
                mlb.classes_[idx],
                round(display_score, 2)
            )
        )

    return predictions
