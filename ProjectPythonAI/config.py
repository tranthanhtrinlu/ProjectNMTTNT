# ai-service/config.py
from pathlib import Path

# Base
BASE_DIR = Path(__file__).parent

# Dataset
DATASET_DIR = BASE_DIR / "datasets"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

# Model
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

EFFICIENTNET_MODEL_PATH = MODELS_DIR / "efficientnet_best.pth"
VIT_MODEL_PATH = MODELS_DIR / "vit_best.pth"


CLASSES = ["FAKE", "REAL"]  


IMAGE_SIZE = 224
BATCH_SIZE = 64
NUM_EPOCHS = 10
LR = 1e-4
DEVICE = "cuda"  
