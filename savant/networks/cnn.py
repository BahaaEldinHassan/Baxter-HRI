from math import sqrt

import torch
import torch.nn as nn
import torch.optim as optim

from savant.settings import NN_DEVICE

from .common import ModelBase, RunnerBase


class CNNModel(ModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        conv_layers = []
        num_layers = 1

        for num_layer in range(num_layers):
            conv_layer = [
                nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
            ]

            if num_layer == num_layers - 1:
                conv_layer.append(nn.AdaptiveMaxPool1d(output_size=self.out_features))
            else:
                conv_layer.append(nn.MaxPool1d(kernel_size=2, stride=2))

            conv_layers.extend(conv_layer)

        self.net = nn.Sequential(
            *conv_layers,
            nn.Flatten(),
            nn.Linear(in_features=32, out_features=self.out_features),
        )

        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.net(torch.unsqueeze(x, 1))

        return self.activation(x)


class CNNRunner(RunnerBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.model = CNNModel(in_features=self.in_features, out_features=self.out_features).to(NN_DEVICE)

        # Loss function
        self.criterion = nn.BCELoss().to(NN_DEVICE)

        # Optimizer
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)

    def train_step(self, features, target_outputs):
        calculated_outputs = self.model(features)  # Add a channel dimension for input data
        # print(features.shape, target_outputs.shape, target_outputs.unsqueeze(1).shape, calculated_outputs.shape)
        loss = self.criterion(calculated_outputs, target_outputs.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
