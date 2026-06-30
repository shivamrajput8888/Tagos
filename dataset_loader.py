import pandas as pd
import ast
import joblib
import torch

from torch.utils.data import Dataset
from transformers import DistilBertTokenizer

from config import (
    MODEL_NAME,
    MAX_LENGTH
)

print("Loading tokenizer...")

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

print("Loading dataset...")

df = pd.read_csv("dataset/processed/final_dataset.csv")

mlb = joblib.load("saved_model/label_encoder.pkl")

df["tags"] = df["tags"].apply(ast.literal_eval)

labels = mlb.transform(df["tags"])


class BlogDataset(Dataset):

    def __init__(self, dataframe, labels):

        self.texts = dataframe["content"].tolist()

        self.labels = labels

    def __len__(self):

        return len(self.texts)

    def __getitem__(self, idx):

        encoding = tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        return {

            "input_ids": encoding["input_ids"].squeeze(),

            "attention_mask": encoding["attention_mask"].squeeze(),

            "labels": torch.tensor(
                self.labels[idx],
                dtype=torch.float
            )
        }


dataset = BlogDataset(df, labels)

print("\nDataset Ready!")

print("Total Samples:", len(dataset))

sample = dataset[0]

print("\nInput Shape :", sample["input_ids"].shape)

print("Attention Shape :", sample["attention_mask"].shape)

print("Label Shape :", sample["labels"].shape)