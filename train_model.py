import ast
import joblib
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from transformers import DistilBertTokenizer

from dataset_loader import BlogDataset
from bert_model import TagosModel

from config import *

print("=" * 50)
print("TAGOS TRAINING")
print("=" * 50)

device = torch.device("cpu")

print("Using Device :", device)

df = pd.read_csv(FINAL_DATASET)

df["tags"] = df["tags"].apply(ast.literal_eval)

df = df.iloc[:TRAIN_SAMPLES]

print("Training Dataset Size :", len(df))
print("TRAIN_SAMPLES :", TRAIN_SAMPLES)

mlb = joblib.load("saved_model/label_encoder.pkl")

labels = mlb.transform(df["tags"])

train_df, val_df, train_labels, val_labels = train_test_split(
    df,
    labels,
    test_size=0.2,
    random_state=RANDOM_STATE
)

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

train_dataset = BlogDataset(
    train_df.reset_index(drop=True),
    train_labels
)

val_dataset = BlogDataset(
    val_df.reset_index(drop=True),
    val_labels
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE
)

model = TagosModel(
    MODEL_NAME,
    TOP_K_TAGS
)

model.to(device)

print()

print("Training Samples :", len(train_dataset))

print("Validation Samples :", len(val_dataset))

print()

print("Model Ready Successfully!")

import torch.nn as nn

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

criterion = nn.BCEWithLogitsLoss()

print()

print("Optimizer Ready")

print("Loss Function Ready")

print()

print("=" * 50)

print("Starting Training")

print("=" * 50)

for epoch in range(EPOCHS):

    print()

    print(f"Epoch {epoch+1}/{EPOCHS}")

    model.train()

    running_loss = 0

    for batch in train_loader:

        input_ids = batch["input_ids"].to(device)

        attention_mask = batch["attention_mask"].to(device)

        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids,
            attention_mask
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print("Training Loss :", running_loss / len(train_loader))

print()

print("Training Finished Successfully")

import os

os.makedirs("saved_model", exist_ok=True)

torch.save(
    model.state_dict(),
    "saved_model/tagos_model.pth"
)

tokenizer.save_pretrained("saved_model")

joblib.dump(
    mlb,
    "saved_model/label_encoder.pkl"
)

print("Model Saved Successfully!")