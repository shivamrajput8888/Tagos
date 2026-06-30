# ==========================================
# TAGOS Configuration
# ==========================================

RAW_DATASET = "dataset/raw/medium_articles.csv"
CLEAN_DATASET = "dataset/processed/clean_articles.csv"
FINAL_DATASET = "dataset/processed/final_dataset.csv"

TOP_K_TAGS = 200

MODEL_NAME = "distilbert-base-uncased"

MAX_LENGTH = 256

TRAIN_SAMPLES = 3000

BATCH_SIZE = 8

EPOCHS = 2

LEARNING_RATE = 2e-5

RANDOM_STATE = 42

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "models") + "/"
SAVE_DIR = os.path.join(BASE_DIR, "saved_model") + "/"