from pathlib import Path

LABELS = ["handshake", "point", "thumbs_down", "thumbs_up", "victory"]

PACKAGE_ROOT_DIR = Path(__file__).parent

TRAINING_DATA_DIR = PACKAGE_ROOT_DIR / "training_data"
