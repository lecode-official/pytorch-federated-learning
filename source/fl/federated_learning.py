"""Implementation of the federated learning protocol and federated averaging (FedAvg)."""

import logging
import collections
from enum import Enum
from typing import Union

import torch

from fl.lifecycle import Trainer, Validator


class AggregationOperator(Enum):
    """Represents an enumeration for the different aggregation operators that can be used by the federated averaging algorithm to aggregate the
    parameters of different neural network layer types.
    """

    MEAN = 'mean'
    SUM = 'sum'


class FederatedAveraging:
    """Represents the federated averaging (FedAvg) algorithm, which can be used to aggregate the models of the federated learning clients into an
    updated global model.
    """

    def __init__(self, global_model: torch.nn.Module) -> None:
        """Initializes a new FederatedAveraging instance.

        Args:
            global_model (torch.nn.Module): The global model of the central server, which is used to determine the correct aggregation operators for
                the parameters of the global model architecture.
        """

        self.parameter_aggregation_operators = self.get_parameter_aggregation_operators(global_model)
        self.reset()

    def reset(self) -> None:
        """Resets the federated averaging algorithm. This is done automatically after updating the global model."""

        self.summed_client_model_parameters = {}
        self.number_of_contributing_clients = 0

    def add_local_model(self, local_model_parameters: collections.OrderedDict) -> None:
        """Adds the specified local model to the new aggregated central server model.

        Args:
            local_model_parameters (collections.OrderedDict): The parameters of the local model of the client that is to be added.
        """

        # Updates the number of clients that have contributed to the aggregated global model, this is later used to average the parameters that have
        # the mean aggregation operator
        self.number_of_contributing_clients += 1

        # Adds the parameters of the specified local model to the aggregation of the parameters that will update the global model
        for parameter_name, _ in self.parameter_aggregation_operators.items():
            client_parameter = local_model_parameters[parameter_name].detach().clone()
            if parameter_name not in self.summed_client_model_parameters:
                self.summed_client_model_parameters[parameter_name] = client_parameter
            else:
                self.summed_client_model_parameters[parameter_name] += client_parameter

    def update_global_model(self, global_model: torch.nn.Module) -> None:
        """Updates the global model of the federated learning central server with the aggregated weights of the clients.

        Args:
            global_model (torch.nn.Module): The global model of the federated learning central server, which is to be updated from the client models.
        """

        global_model_parameters = dict(global_model.named_parameters())
        for name, parameter in self.summed_client_model_parameters.items():
            if self.parameter_aggregation_operators[name] == AggregationOperator.MEAN:
                global_model_parameters[name].data = parameter.data.clone() / self.number_of_contributing_clients
            else:
                global_model_parameters[name].data = parameter.data.clone()

        self.reset()

    def get_parameter_aggregation_operators(self, module: torch.nn.Module, parent_name: str = None) -> dict[str, str]:
        """Different neural network layer types need to be handled differently when aggregating client models into a new global model. For example,
        the weights and biases of linear layers must be averaged, while the number of tracked batches in a BatchNorm layer have to summed up. This
        method goes through all layers (called modules in PyTorch) and determines the operator by which their parameters can be aggregated. Since some
        modules contain other modules themselves, the method goes through all modules recursively.

        Args:
            module (torch.nn.Module): The module for which the aggregation operator of their child modules have to be determined.
            parent_name (str, optional): The name of the parent module. When calling this method on a neural network model, nothing needs to be
                specified. This parameter is only used when by the method itself, when it goes through child modules recursively. Defaults to None.

        Raises:
            ValueError: When a layer type (module) is detected, which is not supported by this implementation of federated averaging, then an
                exception is raised. This indicates that the aggregation operator for the parameters of this module kind still needs to be
                implemented.

        Returns:
            dict[str, str]: Returns a dictionary, which maps the name of a parameter (which is also the exact name of the parameter in the state
                dictionary of the model) to the operator that must be used to aggregate this parameter.
        """

        # Initializes the dictionary that maps the aggregation operator for each parameter of the module
        parameter_aggregation_operators = {}

        # Creates a tuple containing all supported module types, which have no parameters
        module_types_without_parameters = (
            torch.nn.Flatten,
            torch.nn.Unflatten,
            torch.nn.ReLU,
            torch.nn.Sigmoid,
            torch.nn.LeakyReLU,
            torch.nn.MaxPool2d,
            torch.nn.LogSoftmax,
            torch.nn.AdaptiveAvgPool2d,
            torch.nn.Dropout
        )

        # Cycles through the child modules of the specified module to determine the aggregation operator that must be used for their parameters
        for child_name, child_module in module.named_children():

            # Composes the name of the current module (which corresponds to the name of the module in the state dictionary of the model)
            child_name = child_name if parent_name is None else f'{parent_name}.{child_name}'

            # For different module types, different operators are needed to aggregate their parameters
            if isinstance(child_module, torch.nn.Sequential):

                # Sequential modules contains other modules, which are invoked in order, sequential modules do not have any parameters of their own,
                # so the aggregation operator for the parameters of their child modules need to be determined recursively
                parameter_aggregation_operators |= self.get_parameter_aggregation_operators(child_module, parent_name=child_name)

            elif isinstance(child_module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.ConvTranspose2d)):

                # Linear layers, convolutional layers, and transpose convolutional layers have a weight and a bias parameter, which can be aggregated
                # by averaging them
                parameter_aggregation_operators[f'{child_name}.weight'] = AggregationOperator.MEAN
                if child_module.bias is not None:
                    parameter_aggregation_operators[f'{child_name}.bias'] = AggregationOperator.MEAN

            elif isinstance(child_module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):

                # BatchNorm layers have a gamma and a beta parameter (the parameters are called 'weight' and 'bias' respectively), a running mean, and
                # a running variance, which can be aggregated by averaging them, they track the number of batches that they have processed so far,
                # which can be aggregated by summation
                if 'bias' in child_module.__dict__ and child_module.bias:
                    parameter_aggregation_operators[f'{child_name}.weight'] = AggregationOperator.MEAN
                    parameter_aggregation_operators[f'{child_name}.bias'] = AggregationOperator.MEAN
                parameter_aggregation_operators[f'{child_name}.running_mean'] = AggregationOperator.MEAN
                parameter_aggregation_operators[f'{child_name}.running_var'] = AggregationOperator.MEAN
                parameter_aggregation_operators[f'{child_name}.num_batches_tracked'] = AggregationOperator.SUM

            elif isinstance(child_module, torch.nn.GroupNorm):

                # GroupNorm layers have a weight and a bias parameter, which can be aggregated by averaging them
                parameter_aggregation_operators[f'{child_name}.weight'] = AggregationOperator.MEAN
                parameter_aggregation_operators[f'{child_name}.bias'] = AggregationOperator.MEAN

            elif isinstance(child_module, torch.nn.Embedding):

                # Embedding layers have a weight parameter, which can be aggregated by averaging
                parameter_aggregation_operators[f'{child_name}.weight'] = AggregationOperator.MEAN

            elif isinstance(child_module, module_types_without_parameters):

                # These layers have no parameters, therefore, nothing needs to be done
                continue

            else:

                # Since this current module type is not supported, yet, an exception is raised
                raise ValueError(f'The module {child_name} of type {type(child_module)} is not supported by the federated averaging.')

        # Returns the parameter aggregation operators that were determined
        return parameter_aggregation_operators


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
            momentum (float, optional): The momentum of the optimizer. Defaults to 0.9.
            weight_decay (float, optional): The rate at which the weights are decayed during optimization. Defaults to 0.0005.
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

        # Initializes the federated averaging algorithm
        self.model_aggregation_strategy = FederatedAveraging(self.global_model)

    def train_clients_and_update_global_model(self, number_of_local_epochs: int) -> None:
        """Updates the local models of the clients using the parameters of the global model and instructs the clients to train their updated local
        model on their local private training data for the specified number of epochs. Then the global model of the central server is updated by
        aggregating the parameters of the local models of the clients.

        Args:
            number_of_local_epochs (int): The number of epochs for which the clients should train the model on their local data.
        """

        # Cycles through all clients, sends them the global model, and instructs them to train their updated local models on their local data
        global_model_parameters = self.global_model.state_dict()
        for index, client in enumerate(self.clients):
            self.logger.info('Training client %d', index + 1)
            training_loss, training_accuracy, local_model_parameters = client.train(global_model_parameters, number_of_local_epochs)
            self.logger.info('Finished training client %d, Training loss: %f, training accuracy %f', index + 1, training_loss, training_accuracy)
            self.model_aggregation_strategy.add_local_model(local_model_parameters)

        # Updates the parameters of the global model by aggregating the updated parameters of the clients using federated averaging (FedAvg)
        self.model_aggregation_strategy.update_global_model(self.global_model)

    def validate(self) -> tuple[float, float]:
        """Validates the global model of the central server.

        Returns:
            tuple[float, float]: Returns the validation loss and the validation accuracy of the global model.
        """

        return self.validator.validate()

    def save_checkpoint(self, output_file_path: str) -> None:
        """Saves the current state of the global model to a file.

        Args:
            output_path: The path to the file into which the model is to be saved.
        """

        torch.save(self.global_model.state_dict(), output_file_path)
