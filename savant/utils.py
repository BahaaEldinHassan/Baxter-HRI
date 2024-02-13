import numpy as np
import torch
from sklearn.preprocessing import MultiLabelBinarizer

from .settings import NN_DEVICE


def to_tensors(*args, dtype=torch.float):
    # Convert data to PyTorch tensors
    return [torch.tensor(item).to(device=NN_DEVICE, dtype=dtype) for item in args]


def one_hot_encode_classes(class_names):
    """
    Encode a list of class names into one-hot encoded representations.

    Parameters:
    - class_names (list): List of class names

    Returns:
    - classes (dict): Dictionary mapping class names to their one-hot encoded representations
    """
    # Create dictionary to store one-hot encoded representations
    classes = {}

    # Iterate over class names
    for idx, class_name in enumerate(class_names):
        # Create one-hot encoded array
        one_hot_encoded = np.zeros(len(class_names))
        one_hot_encoded[idx] = 1

        # Store one-hot encoded array in dictionary
        classes[class_name] = one_hot_encoded.tolist()
    return classes
