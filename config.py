import torch
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# --- HuggingFace Configuration ---
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

# ---  Model Configuration ---
VIDEO_MODEL_NAME = "microsoft/swin-base-patch4-window7-224-in22k"
AUDIO_MODEL_NAME = "facebook/hubert-large-ls960-ft"
PROJECTION_DIM = 1024
LATENT_DIM = 1024
NUM_LATENTS = 128
NUM_CLASSES = 6  # Example: angry, happy, sad, neutral, fear, disgust

# --- Dataset Configuration ---
# Paths to the datasets
CREMA_D_PATH = "data"
MSP_IMPROV_PATH = "data/MSP-IMPROV"

# Preprocessing
SAMPLE_RATE = 16000
IMAGE_SIZE = 224
MAX_FRAMES = 100  # Max frames to use from a video
MAX_AUDIO_LEN = SAMPLE_RATE * 5  # 5 seconds

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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

NUM_WORKERS = 4 

USE_AMP = True                   # Mixed precision (faster on GPU)
GRAD_ACCUM_STEPS = 1            # Gradient accumulation
EARLY_STOPPING_PATIENCE = 5     # Stop if no improvement
EARLY_STOPPING_MIN_DELTA = 0.001

# Modality Dropout
MODALITY_DROPOUT_RATE = 0.2  # 20% for each modality

# --- Evaluation Configuration ---
EVAL_BATCH_SIZE = 32
CHECKPOINT_PATH = "trained_models/best_model.pth"

# --- File Paths ---
LOG_DIR = "logs/"
MODEL_SAVE_DIR = "trained_models/"