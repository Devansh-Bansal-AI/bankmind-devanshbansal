import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = os.path.join(BASE_DIR, "data", "bank-full.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Feature Definitions
TARGET_COL = "y"

NUMERIC_FEATURES = [
    "age", "balance", "day", "duration", "campaign", "pdays", "previous"
]

CATEGORICAL_FEATURES = [
    "job", "marital", "education", "default", "housing", "loan", 
    "contact", "month", "poutcome"
]

# Evaluation parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2