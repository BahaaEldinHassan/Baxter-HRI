import os
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch.utils.data import DataLoader, TensorDataset
from wisdom.settings import TRAINING_DATA_DIR

from .utils import one_hot_encode_classes, to_tensors


class Dataset:
    def __init__(self, **kwargs):
        self.out_features = None
        self.in_features = None
        self.gesture_recorder_class = "HandGestureRecorder"
        self.label_mappings = None

    def get_label_mappings(self):
        return self.label_mappings

    def _make_dataset(self):
        gesture_recorder_class_found = False

        for item in TRAINING_DATA_DIR.glob("*"):
            if item.is_dir() and item.name == self.gesture_recorder_class:
                gesture_recorder_class_found = True

        if not gesture_recorder_class_found:
            raise RuntimeError(f"No data for gesture recorder class: {self.gesture_recorder_class}")

        X = []
        y = []

        for item in (TRAINING_DATA_DIR / self.gesture_recorder_class).glob("*"):
            if item.is_dir():
                for inner_item in item.glob("*.npy"):
                    flattened_features = np.load(inner_item)

                    # Reshape the flat array into a 2D array.
                    X.append(flattened_features.reshape(-1, 2))

                    y.append(item.name)

        class_names = list(set(y))

        self.out_features = len(class_names)

        encoded_classes = one_hot_encode_classes(class_names)

        self.label_mappings = encoded_classes

        for i in range(len(y)):
            y[i] = encoded_classes[y[i]]

        return np.array(X), np.array(y)

    def _lazy_initialization(self):
        self._make_dataset()

    def process(self):
        # self._lazy_initialization()

        X, y = self._make_dataset()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.4)

        X_train, X_validation, X_test = to_tensors(X_train, X_validation, X_test)

        y_train, y_validation, y_test = to_tensors(
            y_train,
            y_validation,
            y_test,
            dtype=torch.float,
        )

        train_loader, validation_loader, test_loader = self.make_dataloaders(
            (X_train, y_train, {"batch_size": 64, "shuffle": True}),
            (X_validation, y_validation, {"batch_size": 64, "shuffle": True}),
            (X_test, y_test, {"batch_size": 64, "shuffle": False}),
        )

        return train_loader, validation_loader, test_loader

    @staticmethod
    def make_dataloaders(*Xy_pairs):
        # create data loaders
        dataloaders = []

        for x, y, loader_params in Xy_pairs:
            dataset = TensorDataset(x, y)
            dataloader = DataLoader(dataset=dataset, **loader_params)

            dataloaders.append(dataloader)

        return dataloaders
