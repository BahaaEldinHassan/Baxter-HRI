from pathlib import Path

LABELS = [
    "beckoning",  # come here
    # "crossed_fingers",
    "closed_fist",
    "finger_gun",
    # "finger_heart",
    "handshake",
    "okay",
    "open_palm",
    # "point",  # Later
    "three_finger_salute",
    "thumbs_down",
    "thumbs_up",
    # "victory",
]

PACKAGE_ROOT_DIR = Path(__file__).parent

TRAINING_DATA_DIR = PACKAGE_ROOT_DIR / "training_data"
