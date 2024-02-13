import os
from pathlib import Path

import torch

NN_DEVICE = "cpu"

if torch.cuda.is_available():
    NN_DEVICE = "cuda"
elif torch.backends.mps.is_available():
    NN_DEVICE = "mps"
