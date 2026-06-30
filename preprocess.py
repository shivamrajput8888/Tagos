import pandas as pd

# ==========================================
# TAGOS - Data Preprocessing
# ==========================================

INPUT_PATH = "dataset/raw/medium_articles.csv"
OUTPUT_PATH = "dataset/processed/clean_articles.csv"

print("\nLoading dataset...")

df = pd.read_csv(INPUT_PATH)

print(f"Original Shape : {df.shape}")

# ------------------------------------------
# Keep only required columns
# ------------------------------------------

df = df[["title", "text", "tags"]]

# ------------------------------------------
# Fill missing titles
# ------------------------------------------

df["title"] = df["title"].fillna("")

# ------------------------------------------
# Remove rows having missing text or tags
# ------------------------------------------

df = df.dropna(subset=["text", "tags"])

# ------------------------------------------
# Merge title and text
# ------------------------------------------

df["content"] = df["title"] + ". " + df["text"]

# ------------------------------------------
# Keep only final columns
# ------------------------------------------

df = df[["content", "tags"]]

print(f"Processed Shape : {df.shape}")

print("\nFirst 5 rows:\n")
print(df.head())

# ------------------------------------------
# Save cleaned dataset
# ------------------------------------------

df.to_csv(OUTPUT_PATH, index=False)

print("\nClean dataset saved successfully!")
print("Location:", OUTPUT_PATH)
print("\n==============================")
print("Sample Tags")
print("==============================")

for i in range(10):
    print(df["tags"].iloc[i])