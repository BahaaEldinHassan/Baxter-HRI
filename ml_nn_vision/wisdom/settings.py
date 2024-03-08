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
    "point",
    # "three_finger_salute",
    "thumbs_down",
    "thumbs_up",
    # "victory",

    "pose1",
    "pose2",
]

PACKAGE_ROOT_DIR = Path(__file__).parent

YOLO_ROOT_DIR = PACKAGE_ROOT_DIR.parent / "yolo"

TRAINING_DATA_DIR = PACKAGE_ROOT_DIR / "training_data"
