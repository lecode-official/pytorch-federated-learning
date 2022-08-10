"""Implementation of the federated learning protocol and federated averaging (FedAvg)."""

import logging
import collections
from typing import Union

import torch

from source.fl.lifecycle import Trainer, Validator


class FederatedLearningClient:
    """Represents a federated learning client that receives model parameters from the central server and trains it on its local dataset."""

    def __init__(
            self,
            device: Union[str, torch.device],
            local_model: torch.nn.Module,
            local_training_subset: torch.utils.data.Dataset,
            learning_rate: float = 0.01,
            momentum: float = 0.9,
            weight_decay: float = 0.0005,
            batch_size: int = 128) -> None:
        """Initializes a new FederatedLearningClient instance.

        Args:
            device (Union[str, torch.device]): The device on which the local model is to be trained.
            local_model (torch.nn.Module): The local model that is to be trained.
            local_training_subset (torch.utils.data.Dataset): The training subset of the local dataset on which the model is to be trained.
            learning_rate (float, optional): The learning rate of the optimizer. Defaults to 0.01.
            momentum (float, optional): The momentum of the optimizer. Defaults to 0.0005.
            weight_decay (float, optional): The rate at which the weights are decayed during optimization. Defaults to 0.96.
            batch_size (int, optional): The size of mini-batches that are to be used during training. Defaults to 128.
        """

        # Stores the arguments for later use
        self.device = device
        self.local_model = local_model
        self.local_training_subset = local_training_subset
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.batch_size = batch_size

        # Initializes the trainer for the model
        self.trainer = Trainer(
            self.device,
            self.local_model,
            self.local_training_subset,
            self.learning_rate,
            self.momentum,
            self.weight_decay,
            self.batch_size
        )

    def train(self, global_model_parameters: collections.OrderedDict, number_of_epochs: int) -> tuple[float, float, collections.OrderedDict]:
        """Trains the local model on the local data of the client.

        Args:
            global_model_parameters (collections.OrderedDict): The parameters of the global model of the central server that are used to update the
                parameters of the local model.
            number_of_epochs (int): The number of epochs for which the local model is to be trained.

        Returns:
            tuple[float, float, collections.OrderedDict]: Returns a tuple containing the training loss, the training accuracy, and the updated
                parameters of the local model of the client.
        """

        # Copies the parameters of the global model
        local_model_parameters = self.local_model.state_dict()
        for parameter_name in global_model_parameters:
            local_model_parameters[parameter_name].copy_(global_model_parameters[parameter_name])

        # Trains the local model for the specified amount of epochs and saves the training loss and accuracy
        training_loss = None
        training_accuracy = None
        for _ in range(number_of_epochs):
            training_loss, training_accuracy = self.trainer.train_for_one_epoch()

        # Returns the training loss, training accuracy, and the updated parameters of the local model
        return training_loss, training_accuracy, self.trainer.model.state_dict()


class FederatedLearningCentralServer:
    """Represents a federated learning central server, which coordinates the federated learning of the global model."""

    def __init__(
            self,
            clients: list[FederatedLearningClient],
            device: Union[str, torch.device],
            global_model: torch.nn.Module,
            central_validation_subset: torch.utils.data.Dataset,
            batch_size: int = 128) -> None:
        """Initializes a new FederatedLearningCentralServer instance.

        Args:
            clients (list[FederatedLearningClient]): The federated learning clients.
            device (Union[str, torch.device]): The device on which the global model of the central server is to be validated.
            global_model (torch.nn.Module): The global model of the central server, which is distributed to the federated learning clients for
                training.
            central_validation_subset (torch.utils.data.Dataset): The validation subset on which the global model is to be validated.
            batch_size (int, optional): The size of the mini-batches that are to be used during the validation. Defaults to 128.
        """

        # Stores the arguments for later use
        self.clients = clients
        self.device = device
        self.global_model = global_model
        self.central_validation_subset = central_validation_subset
        self.batch_size = batch_size

        # Initializes the logger
        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)

        # Initializes the validator for the global model
        self.validator = Validator(self.device, self.global_model, self.central_validation_subset, self.batch_size)

    def train_clients_and_update_global_model(self, number_of_local_epochs: int) -> None:
        """Updates the local models of the clients using the parameters of the global model and instructs the clients to train their updated local
        model on their local private training data for the specified number of epochs. Then the global model of the central server is updated by
        aggregating the parameters of the local models of the clients.

        Args:
            number_of_local_epochs (int): The number of epochs for which the clients should train the model on their local data.
        """

        # Cycles through all clients, sends them the global model, and instructs them to train their updated local models on their local data
        client_model_parameters = []
        global_model_parameters = self.global_model.state_dict()
        for index, client in enumerate(self.clients):
            self.logger.info('Training client %d', index + 1)
            training_loss, training_accuracy, local_model_parameters = client.train(global_model_parameters, number_of_local_epochs)
            self.logger.info('Finished training client %d, Training loss: %f, training accuracy %f', index + 1, training_loss, training_accuracy)
            client_model_parameters.append(local_model_parameters)

        # Updates the parameters of the global model by aggregating the updated parameters of the clients using federated averaging (FedAvg)
        summed_client_model_parameters = {}
        for parameters in client_model_parameters:
            for parameter_name in parameters:
                if parameter_name not in summed_client_model_parameters:
                    summed_client_model_parameters[parameter_name] = parameters[parameter_name].clone()
                else:
                    summed_client_model_parameters[parameter_name] += parameters[parameter_name].clone()
        for parameter_name in summed_client_model_parameters:
            global_model_parameters[parameter_name].copy_(summed_client_model_parameters[parameter_name] / len(self.clients))
