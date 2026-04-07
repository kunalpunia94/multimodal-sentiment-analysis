import torch
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# --- HuggingFace Configuration ---
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN", "")

# ---  Model Configuration ---
VIDEO_MODEL_NAME = os.getenv("VIDEO_MODEL_NAME", "microsoft/swin-base-patch4-window7-224-in22k")
AUDIO_MODEL_NAME = os.getenv("AUDIO_MODEL_NAME", "facebook/hubert-large-ls960-ft")
PROJECTION_DIM = int(os.getenv("PROJECTION_DIM", 1024))
LATENT_DIM = int(os.getenv("LATENT_DIM", 1024))
NUM_LATENTS = int(os.getenv("NUM_LATENTS", 128))
NUM_CLASSES = 6  # Example: angry, happy, sad, neutral, fear, disgust

# --- Dataset Configuration ---
# Paths to the datasets
CREMA_D_PATH = "data"
MSP_IMPROV_PATH = "data/MSP-IMPROV"

# Preprocessing
SAMPLE_RATE = 16000
IMAGE_SIZE = 224
MAX_FRAMES = int(os.getenv("MAX_FRAMES", 8))  # Max frames to use from a video
MAX_AUDIO_LEN = SAMPLE_RATE * int(os.getenv("MAX_AUDIO_SEC", 3))  # seconds

# Emotion labels for CREMA-D (example)
CREMA_D_LABELS = {
    "ANG": "angry",
    "HAP": "happy",
    "SAD": "sad",
    "NEU": "neutral",
    "FEA": "fear",
    "DIS": "disgust",
}

EMOTION_MAP = {
    "angry": 0,
    "happy": 1,
    "sad": 2,
    "neutral": 3,
    "fear": 4,
    "disgust": 5,
}

# --- Training Configuration ---
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

BATCH_SIZE = int(os.getenv("BATCH_SIZE", 1))
EPOCHS = int(os.getenv("EPOCHS", 10))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 1e-4))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", 1e-5))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 2))
USE_AMP = os.getenv("USE_AMP", "1") == "1"
GRAD_ACCUM_STEPS = int(os.getenv("GRAD_ACCUM_STEPS", 1))
EARLY_STOPPING_PATIENCE = int(os.getenv("EARLY_STOPPING_PATIENCE", 3))
EARLY_STOPPING_MIN_DELTA = float(os.getenv("EARLY_STOPPING_MIN_DELTA", 1e-4))

# Modality Dropout
MODALITY_DROPOUT_RATE = 0.2  # 20% for each modality

# --- Evaluation Configuration ---
EVAL_BATCH_SIZE = 32
CHECKPOINT_PATH = "trained_models/best_model.pth"

# --- File Paths ---
LOG_DIR = "logs/"
MODEL_SAVE_DIR = "trained_models/"
