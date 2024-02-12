import abc
import csv
import time
import typing

import matplotlib.pyplot as plt
import numpy
import pandas
import torch
import torch.nn as nn

from savant.settings import NN_DEVICE
from savant.utils import to_tensors


class ModelBase(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        print("Initializing Model:", self.__class__.__name__)

        self.in_features = kwargs["in_features"]
        self.out_features = kwargs["out_features"]


class RunnerBase(abc.ABC):
    def __init__(self, **kwargs):
        print("Initializing Runner:", self.__class__.__name__)

        self.learning_rate = kwargs["learning_rate"]
        self.momentum = 0.9  # kwargs["momentum"]
        self.num_epochs = kwargs["num_epochs"]
        self.current_epoch = 0
        self.in_features = kwargs["in_features"]
        self.out_features = kwargs["out_features"]
        self.num_gates = kwargs.get("num_gates", 1)

        self.model: typing.Optional[ModelBase] = None
        self.criterion: typing.Optional[nn.Module] = None
        self.mse_criterion = nn.MSELoss().to(NN_DEVICE)
        self.optimizer: typing.Optional[nn.Module] = None

        self.epoch_losses = {
            "training": [],
            "validation": [],
        }

    # def evaluate(self, dataloader):
    #     # Calculate validation loss.
    #     self.model.eval()
    #
    #     total_mse = 0.0
    #     total_spe = 0.0
    #     num_samples = len(dataloader.dataset)
    #
    #     with torch.no_grad():
    #         for features, target_outputs in dataloader:
    #             calculated_outputs = self.model(features)
    #
    #             # Keep adding RMSE
    #             # NOTE: .item() returns scalar with total loss.
    #             total_mse += self.mse_criterion(calculated_outputs, target_outputs).item()
    #
    #             # RMSPE
    #             total_spe += calculate_total_squared_percentage_error(calculated_outputs, target_outputs)
    #
    #     # Calculate average RMSE and RMSPE
    #     rmse = numpy.sqrt(total_mse / num_samples)
    #     rmspe = numpy.sqrt(total_spe / num_samples)
    #
    #     print(f"RMSE: {rmse:.10f}")
    #     print(f"RMSPE: {rmspe:.10f}")

    @staticmethod
    def get_best_epoch(epoch_losses, default):
        try:
            return epoch_losses.index(min(epoch_losses))
        except ValueError:
            return default

    def load_state(self, state_filename):
        checkpoint = torch.load(state_filename)

        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.criterion.load_state_dict(checkpoint["criterion_state"])

        self.num_epochs = checkpoint["num_epochs"]
        self.current_epoch = checkpoint["current_epoch"]

        self.epoch_losses = checkpoint["epoch_losses"]

    def save_loss_plot(self, target_filename="loss_plot.png"):
        # Extract losses for training and validation.
        training_losses = self.epoch_losses["training"]
        best_training_epoch_index = self.get_best_epoch(training_losses, self.current_epoch)
        validation_losses = self.epoch_losses["validation"]
        best_validation_epoch_index = self.get_best_epoch(validation_losses, self.current_epoch)

        # Set the figure size to 1000 x 600 pixels.
        plt.figure(figsize=(10, 6))

        # Create training line plot.
        plt.plot(range(1, len(training_losses) + 1), training_losses, color="blue", label="Training Loss")
        plt.axvline(
            x=best_training_epoch_index + 1,
            color="blue",
            label=f"Best Training Epoch: {best_training_epoch_index + 1}",
        )

        # Create validation line plot.
        plt.plot(range(1, len(validation_losses) + 1), validation_losses, color="orange", label="Validation Loss")
        plt.axvline(
            x=best_validation_epoch_index + 1,
            color="orange",
            label=f"Best Validation Epoch: {best_validation_epoch_index + 1}",
        )

        # Set labels and title.
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss Over Epochs\n" f"Learning Rate: {self.learning_rate}")

        # Show the legend.
        plt.legend()

        # Save the plot to an image file.
        plt.savefig(target_filename)

    # def predict(self, test_features):
    #     self.model.eval()
    #
    #     predictions = {}
    #
    #     with torch.no_grad():
    #         for features in test_features:
    #             store_id = features[0]
    #
    #             if not numpy.isnan(store_id):
    #                 scaled_features = self.feature_scaler.transform(features[1:].reshape(1, -1))
    #                 calculated_outputs = self.model(to_tensors(scaled_features)[0])
    #
    #                 calculated_output = float(
    #                     self.output_scaler.inverse_transform(tensor_to_numpy(calculated_outputs)).reshape(-1)
    #                 )
    #
    #                 if numpy.isnan(calculated_output) or (calculated_output < 0):
    #                     calculated_output = 0.0
    #
    #                 predictions[int(store_id)] = calculated_output
    #
    #     df = pandas.DataFrame(list(predictions.items()), columns=["Id", "Sales"])
    #
    #     df.to_csv(DATA_ROOT_DIR / "final_submission.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)

    def save_state(self, state_filename):
        torch.save(
            {
                "current_epoch": self.current_epoch,
                "epoch_losses": self.epoch_losses,
                "num_epochs": self.num_epochs,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "criterion_state": self.criterion.state_dict(),
            },
            state_filename,
        )

    @abc.abstractmethod
    def train_step(self, features, target_outputs):
        ...

    def validate(self, dataloader):
        # Calculate validation loss.
        self.model.eval()

        total_validation_loss = 0.0
        num_samples = len(dataloader.dataset)

        with torch.no_grad():
            for features, target_outputs in dataloader:
                calculated_outputs = self.model(features)

                total_validation_loss += self.criterion(calculated_outputs, target_outputs.unsqueeze(1)).item()

        validation_loss_avg = total_validation_loss / num_samples

        return validation_loss_avg

    def train(self, train_loader, validation_loader):
        num_samples = len(train_loader.dataset)

        while self.current_epoch < self.num_epochs:
            epoch_start_time = time.time()

            total_training_loss = 0.0

            self.model.train()

            for features, target_outputs in train_loader:
                loss = self.train_step(features, target_outputs)

                total_training_loss += loss.item()

            training_loss_avg = total_training_loss / num_samples

            validation_loss_avg = self.validate(validation_loader)

            seconds_elapsed = time.time() - epoch_start_time

            self.epoch_losses["training"].append(training_loss_avg)
            self.epoch_losses["validation"].append(validation_loss_avg)

            try:
                best_epoch_index = self.epoch_losses["training"].index(min(self.epoch_losses["training"]))
            except ValueError:
                best_epoch_index = self.current_epoch

            print(
                f"[{self.current_epoch+1}/{self.num_epochs}] "
                f"Training loss: {training_loss_avg:.10f} | "
                f"Validation loss: {validation_loss_avg:.10f} | "
                f"Best training epoch: {best_epoch_index+1} "
                f"({seconds_elapsed:.2f}s)"
            )

            self.current_epoch += 1
