import pandas as pd
import ast
import joblib

from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer

from config import CLEAN_DATASET, TOP_K_TAGS, MODEL_DIR

print("Loading cleaned dataset...")

df = pd.read_csv(CLEAN_DATASET)

# Convert string to Python list
df["tags"] = df["tags"].apply(ast.literal_eval)

# ------------------------------------
# Count tag frequencies
# ------------------------------------

counter = Counter()

for tags in df["tags"]:
    counter.update(tags)

top_tags = [tag for tag, count in counter.most_common(TOP_K_TAGS)]

print(f"\nKeeping Top {TOP_K_TAGS} Tags")

# ------------------------------------
# Keep only top tags
# ------------------------------------

df["tags"] = df["tags"].apply(
    lambda tags: [tag for tag in tags if tag in top_tags]
)

# Remove rows having no remaining tags
df = df[df["tags"].map(len) > 0]

print("\nDataset after filtering:", df.shape)

# ------------------------------------
# Multi Label Encoding
# ------------------------------------

mlb = MultiLabelBinarizer(classes=top_tags)

labels = mlb.fit_transform(df["tags"])

print("\nEncoded Label Shape:")

print(labels.shape)

# ------------------------------------
# Save Encoder
# ------------------------------------

joblib.dump(mlb, MODEL_DIR + "label_encoder.pkl")

# Save filtered dataset
df.to_csv("dataset/processed/final_dataset.csv", index=False)

print("\nSaved Successfully!")