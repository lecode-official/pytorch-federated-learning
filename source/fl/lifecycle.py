"""Lifecycle primitives for neural network models, including training and validation."""

import logging
from typing import Union

import torch
import psutil
import torchmetrics
from tqdm import tqdm


class Trainer:
    """Trains a neural network model."""

    def __init__(
            self,
            device: Union[str, torch.device],
            model: torch.nn.Module,
            training_subset: torch.utils.data.Dataset,
            learning_rate: float = 0.01,
            momentum: float = 0.9,
            weight_decay: float = 0.0005,
            batch_size: int = 128) -> None:
        """Initializes a new Trainer instance.

        Args:
            device (Union[str, torch.device]): The device on which the model is to be trained.
            model (torch.nn.Module): The model that is to be trained.
            training_subset (torch.utils.data.Dataset): The training subset of the dataset on which the model is to be trained.
            learning_rate (float, optional): The learning rate of the optimizer. Defaults to 0.01.
            momentum (float, optional): The momentum of the optimizer. Defaults to 0.0005.
            weight_decay (float, optional): The rate at which the weights are decayed during optimization. Defaults to 0.96.
            batch_size (int, optional): The size of mini-batches that are to be used during training. Defaults to 128.
        """

        # Stores the arguments for later use
        self.device = device
        self.model = model
        self.training_subset = training_subset
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.batch_size = batch_size

        # Initializes the logger
        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)

        # Initializes a flag, which is set when the training should be aborted
        self.is_aborting = False

        # Makes sure that the model is on the specified device
        self.model = self.model.to(self.device)

        # Creates the data loaders
        self.training_data_loader = torch.utils.data.DataLoader(
            self.training_subset,
            batch_size=self.batch_size,
            num_workers=psutil.cpu_count(logical=False),
            shuffle=True,
            drop_last=False,
            pin_memory=True
        )

        # Creates the loss function
        self.loss_function = torch.nn.CrossEntropyLoss().to(self.device)

        # Creates the optimizer for the training (Adam is generally a good choice, I have tried SGD, but it was worse than Adam)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            nesterov=True
        )

    def train_for_one_epoch(self) -> tuple[float, float]:
        """Trains the model for one epoch

        Returns:
            tuple[float, float]: Returns the training loss and the training accuracy.
        """

        # Parts of the model might behave differently during training as compared to inference, therefore, the model is switched to training mode
        self.model.train()

        # Initializes the loss and accuracy metrics
        accuracy = torchmetrics.Accuracy().to(self.device)
        mean_loss = torchmetrics.MeanMetric().to(self.device)

        # Cycles through the entire dataset and trains the model on the samples
        for inputs, targets in tqdm(self.training_data_loader, desc=f'Training', unit='iterations'):

            # Resets the gradients of the optimizer (otherwise the gradients would just accumulate)
            self.optimizer.zero_grad()

            # Moves the inputs and the targets to the selected device
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Performs a forward pass through the neural network and computes the loss
            predictions = self.model(inputs)
            loss = self.loss_function(predictions, targets)

            # Updates the training metrics
            mean_loss.update(loss)
            accuracy.update(predictions, targets)

            # Performs the backward path and the optimization step
            loss.backward()
            self.optimizer.step()

            # If the user hit Ctrl+C, then the training of the model is aborted
            if self.is_aborting:
                self.logger.info('Aborting training... Hit Ctrl+C again to force quit...')
                break

        # Returns the training loss and accuracy
        return mean_loss.compute().cpu().numpy().item(), accuracy.compute().cpu().numpy().item()

    def change_learning_rate(self, new_learning_rate: float) -> None:
        """Changes the learning rate of the optimizer.

        Args:
            new_learning_rate (float): The new learning rate that is to be used by the optimizer.
        """

        for parameter_group in self.optimizer.param_groups:
            parameter_group['lr'] = new_learning_rate

    def abort_training(self) -> None:
        """Graciously aborts the training."""

        self.is_aborting = True

    def save_checkpoint(self, output_file_path: str) -> None:
        """Saves the current state of the model to a checkpoint file.

        Args:
            output_path: The path to the file into which the model is to be saved.
        """

        torch.save(self.model.state_dict(), output_file_path)


class Validator:
    """Validates a neural network model."""

    def __init__(
            self,
            device: Union[str, torch.device],
            model: torch.nn.Module,
            validation_subset: torch.utils.data.Dataset,
            batch_size: int = 128) -> None:
        """Initializes a new Validator instance.

        Args:
            device (Union[str, torch.device]): The device on which the validation is to be performed.
            model (torch.nn.Module): The model that is to be validated.
            validation_subset (torch.utils.data.Dataset): The validation subset of the dataset on which the model is to be validated.
            batch_size (int, optional): The size of the mini-batches that are to be used during the validation. Defaults to 128.
        """

        # Stores the arguments for later use
        self.device = device
        self.model = model
        self.validation_subset = validation_subset
        self.batch_size = batch_size

        # Makes sure that the model is on the specified device
        self.model = self.model.to(self.device)

        # Creates the data loaders
        self.validation_data_loader = torch.utils.data.DataLoader(
            self.validation_subset,
            batch_size=self.batch_size,
            num_workers=psutil.cpu_count(logical=False),
            shuffle=False,
            drop_last=False,
            pin_memory=True
        )

        # Creates the loss function
        self.loss_function = torch.nn.CrossEntropyLoss().to(self.device)

    def validate(self) -> tuple[float, float]:
        """Validates the model.

        Returns:
            tuple[float, float]: Returns the validation loss and the validation accuracy of the model.
        """

        # Parts of the model might behave differently during validation as compared to training, therefore, the model is switched to validation mode
        self.model.eval()

        # Since we are only validating the model, the gradient does not have to be computed
        with torch.no_grad():

            # Initializes the loss and accuracy metrics
            mean_loss = torchmetrics.MeanMetric().to(self.device)
            accuracy = torchmetrics.Accuracy().to(self.device)

            # Cycles through the whole validation subset of the dataset and performs the validation
            for inputs, targets in tqdm(self.validation_data_loader, desc='Validating', unit='iterations'):

                # Transfers the batch to the selected device
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                # Performs the forward pass through the neural network and computes the loss
                predictions = self.model(inputs)
                loss = self.loss_function(predictions, targets)

                # Updates the validation metrics
                mean_loss.update(loss)
                accuracy.update(predictions, targets)

            # Computes the validation metrics and returns them
            return mean_loss.compute().cpu().numpy().item(), accuracy.compute().cpu().numpy().item()
