import os
import torch
import joblib

from bert_model import TagosModel
from config import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "saved_model")

MODEL_PATH = os.path.join(SAVE_DIR, "tagos_model.pth")
LABEL_PATH = os.path.join(SAVE_DIR, "label_encoder.pkl")
ONNX_PATH = os.path.join(SAVE_DIR, "tagos_model.onnx")

print("Loading label encoder...")
mlb = joblib.load(LABEL_PATH)

print("Creating model...")
model = TagosModel(
    MODEL_NAME,
    len(mlb.classes_)
)

print("Loading trained weights...")
state_dict = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(state_dict)
model.eval()

print("Creating dummy inputs...")

dummy_input_ids = torch.ones(
    (1, MAX_LENGTH),
    dtype=torch.long
)

dummy_attention_mask = torch.ones(
    (1, MAX_LENGTH),
    dtype=torch.long
)

print("Exporting model to ONNX...")

torch.onnx.export(
    model,
    (
        dummy_input_ids,
        dummy_attention_mask
    ),
    ONNX_PATH,
    input_names=[
        "input_ids",
        "attention_mask"
    ],
    output_names=[
        "logits"
    ],
    dynamic_axes={
        "input_ids": {
            0: "batch_size",
            1: "sequence_length"
        },
        "attention_mask": {
            0: "batch_size",
            1: "sequence_length"
        },
        "logits": {
            0: "batch_size"
        }
    },
    opset_version=17,
    do_constant_folding=True
)

print("ONNX conversion complete!")
print(f"Saved to: {ONNX_PATH}")
