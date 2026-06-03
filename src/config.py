"""Central configuration. All paths are cross-platform (pathlib)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
AUG_DIR = ROOT / "aug_data"
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"

for _d in (DATA_DIR, AUG_DIR, MODELS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Image / model
IMG_SIZE = (120, 120)
CAPTURE_RES = (640, 480)          # width, height used when normalising raw labels
PARTITIONS = ("train", "test", "val")
AUG_PER_IMAGE = 60

# Training
EPOCHS = 40
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
CLASS_LOSS_WEIGHT = 0.5

# Detector weights (saved after training)
DETECTOR_WEIGHTS = MODELS_DIR / "facetracker.h5"

# Liveness
EAR_THRESHOLD = 0.21              # eye-aspect-ratio below this = eye closed
EAR_CONSEC_FRAMES = 2            # consecutive closed frames to count a blink
BLINKS_REQUIRED = 2              # blinks needed to pass challenge
