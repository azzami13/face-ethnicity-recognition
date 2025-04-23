import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent.absolute()

# Data directories
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
FACES_DIR = os.path.join(PROCESSED_DATA_DIR, "faces")
ALIGNED_FACES_DIR = os.path.join(PROCESSED_DATA_DIR, "aligned")
EMBEDDINGS_DIR = os.path.join(PROCESSED_DATA_DIR, "embeddings")

# Model directories
MODELS_DIR = os.path.join(BASE_DIR, "models")
FACE_DETECTION_MODEL_DIR = os.path.join(MODELS_DIR, "face_detection")
FACE_EMBEDDING_MODEL_DIR = os.path.join(MODELS_DIR, "face_embedding")
ETHNICITY_MODEL_DIR = os.path.join(MODELS_DIR, "ethnicity_classification")

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FACES_DIR, ALIGNED_FACES_DIR, 
                 EMBEDDINGS_DIR, MODELS_DIR, FACE_DETECTION_MODEL_DIR, 
                 FACE_EMBEDDING_MODEL_DIR, ETHNICITY_MODEL_DIR]:
    os.makedirs(directory, exist_ok=True)

# Data split parameters
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Ethnicity mapping
ETHNICITY_MAPPING = {
    "Jawa": 0,
    "Sunda": 1,
    "Batak": 2,
    "Bugis": 3,
    "Minang": 4,
    "Bali": 5,
    "Dayak": 6,
    "Melayu": 7,
    "Aceh": 8,
    "Betawi": 9,
}

ETHNICITY_MAPPING_REVERSE = {v: k for k, v in ETHNICITY_MAPPING.items()}

# Face detection parameters
FACE_DETECTION_CONFIDENCE_THRESHOLD = 0.9
FACE_DETECTION_MARGIN = 0.2  # 20% margin around detected face

# Face similarity parameters
FACE_EMBEDDING_SIZE = 512
FACE_SIMILARITY_THRESHOLD = 0.6  # Threshold for considering same person

# Training parameters
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
RANDOM_SEED = 42