import torch

from .settings import NN_DEVICE


def to_tensors(*args, dtype=torch.float):
    # Convert data to PyTorch tensors
    return [torch.tensor(item).to(device=NN_DEVICE, dtype=dtype) for item in args]
