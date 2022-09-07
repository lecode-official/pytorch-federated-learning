"""Implementation of the federated learning protocol and federated averaging (FedAvg)."""

import random
import logging
import collections
from enum import Enum
from typing import Optional, Union

import numpy
import torch

from fl.lifecycle import Trainer, Validator
from fl.models import ModelType, NormalizationLayerKind, create_model


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

        self.summed_client_model_parameters = {}
        self.number_of_contributing_clients = 0

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

        global_model_parameters = global_model.state_dict()
        for name, parameter in self.summed_client_model_parameters.items():
            if self.parameter_aggregation_operators[name] == AggregationOperator.MEAN:
                global_model_parameters[name].copy_(parameter / self.number_of_contributing_clients)
            else:
                global_model_parameters[name].copy_(parameter)

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
            client_id: int,
            device: Union[str, torch.device],
            local_model_type: ModelType,
            local_model_normalization_layer_kind: NormalizationLayerKind,
            local_training_subset: torch.utils.data.Dataset,
            sample_shape: tuple,
            number_of_classes: int,
            momentum: float = 0.9,
            weight_decay: float = 0.0005,
            batch_size: int = 128) -> None:
        """Initializes a new FederatedLearningClient instance.

        Args:
            client_id (int): An ID, which uniquely identifies the client.
            device (Union[str, torch.device]): The device on which the local model is to be trained.
            local_model_type (ModelType): The type of local model that is to be trained.
            local_model_normalization_layer_kind (NormalizationLayerKind): The kind of the normalization layer that is used for the local model.
            local_training_subset (torch.utils.data.Dataset): The training subset of the local dataset on which the model is to be trained.
            sample_shape (tuple): The shape of the samples in the dataset.
            number_of_classes (int): The number of classes in the dataset.
            momentum (float, optional): The momentum of the optimizer. Defaults to 0.9.
            weight_decay (float, optional): The rate at which the weights are decayed during optimization. Defaults to 0.0005.
            batch_size (int, optional): The size of mini-batches that are to be used during training. Defaults to 128.
        """

        # Stores the arguments for later use
        self.client_id = client_id
        self.device = device
        self.local_model_type = local_model_type
        self.local_model_normalization_layer_kind = local_model_normalization_layer_kind
        self.local_training_subset = local_training_subset
        self.sample_shape = sample_shape
        self.number_of_classes = number_of_classes
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.batch_size = batch_size

        # Initializes the logger
        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)

        # Initializes the property for the trainer
        self.trainer = None

        # Initializes a flag, which is set when the training should be aborted
        self.is_aborting = False

    def train(
            self,
            global_model_parameters: collections.OrderedDict,
            learning_rate: float,
            number_of_epochs: int
        ) -> tuple[float, float, collections.OrderedDict]:
        """Trains the local model on the local data of the client.

        Args:
            global_model_parameters (collections.OrderedDict): The parameters of the global model of the central server that are used to update the
                parameters of the local model.
            learning_rate (float): The current learning rate that is to be used. For each communication round a different learning rate can be used,
                e.g., by decaying the learning rate.
            number_of_epochs (int): The number of epochs for which the local model is to be trained.

        Returns:
            tuple[float, float, collections.OrderedDict]: Returns a tuple containing the training loss, the training accuracy, and the updated
                parameters of the local model of the client.
        """

        # Creates the local model
        local_model = create_model(
            self.local_model_type,
            input_shape=self.sample_shape,
            number_of_classes=self.number_of_classes,
            normalization_layer_kind=self.local_model_normalization_layer_kind
        )

        # Copies the parameters of the global model
        local_model_parameters = local_model.state_dict()
        for parameter_name in global_model_parameters:
            local_model_parameters[parameter_name].copy_(global_model_parameters[parameter_name])

        # Creates the trainer for the model
        self.trainer = Trainer(
            self.device,
            local_model,
            self.local_training_subset,
            learning_rate,
            self.momentum,
            self.weight_decay,
            self.batch_size
        )

        # Trains the local model for the specified amount of epochs and saves the training loss and accuracy
        training_loss = None
        training_accuracy = None
        for _ in range(number_of_epochs):
            if self.is_aborting:
                self.logger.info('Aborting local training of client %d... Hit Ctrl+C again to force quit...', self.client_id)
                break
            training_loss, training_accuracy = self.trainer.train_for_one_epoch()

        # Returns the training loss, training accuracy, and the updated parameters of the local model
        local_model_parameters = self.trainer.model.state_dict()
        self.trainer = None
        return training_loss, training_accuracy, local_model_parameters

    def abort_training(self) -> None:
        """Graciously aborts the federated learning."""

        self.is_aborting = True
        if self.trainer is not None:
            self.trainer.abort_training()


class FederatedLearningCentralServer:
    """Represents a federated learning central server, which coordinates the federated learning of the global model."""

    def __init__(
            self,
            clients: list[FederatedLearningClient],
            number_of_clients_per_communication_round: Optional[int],
            device: Union[str, torch.device],
            global_model_type: ModelType,
            global_model_normalization_layer_kind: NormalizationLayerKind,
            central_validation_subset: torch.utils.data.Dataset,
            sample_shape: tuple,
            number_of_classes: int,
            initial_learning_rate: float = 0.1,
            learning_rate_decay: float = 0.95,
            batch_size: int = 128) -> None:
        """Initializes a new FederatedLearningCentralServer instance.

        Args:
            clients (list[FederatedLearningClient]): The federated learning clients.
            number_of_clients_per_communication_round (int): One of the primary bottlenecks in the communication between the central server and its
                clients is the number of clients that the central server has to communicate with in each communication round. One easy method of
                reducing this overhead, is to subsample the client population. In each communication round, the central server only selects a subset
                of clients, which will train and communicate their updates back. This parameter specifies the number of clients that will be selected
                at random in each communication round. If not specified, this defaults to the number of clients.
            device (Union[str, torch.device]): The device on which the global model of the central server is to be validated.
            global_model_type (ModelType): The type of model that is to be used as global model for the central server.
            global_model_normalization_layer_kind (NormalizationLayerKind): The kind of the normalization layer that is used for the global model.
            central_validation_subset (torch.utils.data.Dataset): The validation subset on which the global model is to be validated.
            sample_shape (tuple): The shape of the samples in the dataset.
            number_of_classes (int): The number of classes in the dataset.
            initial_learning_rate (float, optional): The initial learning rate of the optimizer. Defaults to 0.1.
            learning_rate_decay (float, optional): The learning rate is decayed exponentially during the training. This parameter is the decay rate of
                the learning rate. A decay rate 1.0 would result in no decay at all. Defaults to 0.95.
            batch_size (int, optional): The size of the mini-batches that are to be used during the validation. Defaults to 128.

        Raises:
            ValueError: When the number of clients per communication round is less than 1 or more than the total number of clients, a ValueError is
                raised.
        """

        # If no number of clients per communication round was specified, then it defaults to the number of clients
        if number_of_clients_per_communication_round is None:
            number_of_clients_per_communication_round = len(clients)

        # Validates the arguments
        if number_of_clients_per_communication_round < 1:
            raise ValueError('At least one client must be selected for training in each communication round.')
        if number_of_clients_per_communication_round > len(clients):
            raise ValueError('The number of clients to select for each communication round is greater than the number of clients.')

        # Stores the arguments for later use
        self.clients = clients
        self.number_of_clients_per_communication_round = number_of_clients_per_communication_round
        self.device = device
        self.global_model_type = global_model_type
        self.global_model_normalization_layer_kind = global_model_normalization_layer_kind
        self.central_validation_subset = central_validation_subset
        self.sample_shape = sample_shape
        self.number_of_classes = number_of_classes
        self.initial_learning_rate = initial_learning_rate
        self.current_learning_rate = initial_learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.batch_size = batch_size

        # Initializes the logger
        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)

        # Creates the global model and the validator for it
        self.global_model = create_model(
            self.global_model_type,
            input_shape=self.sample_shape,
            number_of_classes=self.number_of_classes,
            normalization_layer_kind=self.global_model_normalization_layer_kind
        )
        self.validator = Validator(self.device, self.global_model, self.central_validation_subset, self.batch_size)

        # Initializes the federated averaging algorithm
        self.model_aggregation_strategy = FederatedAveraging(self.global_model)

        # Initializes a flag, which is set when the training should be aborted
        self.is_aborting = False

    def train_clients_and_update_global_model(self, number_of_local_epochs: int) -> tuple[list[float], list[float]]:
        """Updates the local models of the clients using the parameters of the global model and instructs the clients to train their updated local
        model on their local private training data for the specified number of epochs. Then the global model of the central server is updated by
        aggregating the parameters of the local models of the clients.

        Args:
            number_of_local_epochs (int): The number of epochs for which the clients should train the model on their local data.

        Returns:
            tuple[list[float], list[float]]: Returns a tuple containing a list of training losses and a list of training accuracies of all clients.
                Clients that did not participate in the current communication round have a loss and an accuracy of numpy.nan.
        """

        # Selects a subsample of the client population for the current communication round
        client_subsample = random.sample(self.clients, self.number_of_clients_per_communication_round)

        # Cycles through all clients, sends them the global model, and instructs them to train their updated local models on their local data
        global_model_parameters = self.global_model.state_dict()
        client_training_losses = [numpy.nan for _ in range(len(self.clients))]
        client_training_accuracies = [numpy.nan for _ in range(len(self.clients))]
        for index, client in enumerate(client_subsample):

            # If the user hit Ctrl+C, then the communication round is aborted
            if self.is_aborting:
                self.logger.info('Aborting communication round... Hit Ctrl+C again to force quit...')
                break

            # Trains the client and reports the training loss and training accuracy of it
            self.logger.info('Training client %d (%d/%d)...', client.client_id, index + 1, self.number_of_clients_per_communication_round)
            training_loss, training_accuracy, local_model_parameters = client.train(
                global_model_parameters,
                self.current_learning_rate,
                number_of_local_epochs
            )
            self.logger.info(
                'Finished training client %d, training loss: %f, training accuracy: %f%%, learning rate: %f',
                client.client_id,
                training_loss,
                training_accuracy * 100,
                self.current_learning_rate
            )

            # Stores the training loss and training accuracy of the client
            client_training_losses[client.client_id - 1] = training_loss
            client_training_accuracies[client.client_id - 1] = training_accuracy

            # Adds the updated parameters of the local model of the client to the aggregated model parameters from which the global model will be
            # updated after the communication round has finished
            self.model_aggregation_strategy.add_local_model(local_model_parameters)

        # Updates the parameters of the global model by aggregating the updated parameters of the clients using federated averaging (FedAvg)
        self.model_aggregation_strategy.update_global_model(self.global_model)

        # Decays the learning rate
        self.current_learning_rate = self.current_learning_rate * self.learning_rate_decay

        # Returns the training losses and training accuracies of all clients (clients that did not participate in the communication round have a loss
        # and accuracy of numpy.nan)
        return client_training_losses, client_training_accuracies

    def validate(self) -> tuple[float, float]:
        """Validates the global model of the central server.

        Returns:
            tuple[float, float]: Returns the validation loss and the validation accuracy of the global model.
        """

        return self.validator.validate()

    def abort_training(self) -> None:
        """Graciously aborts the federated learning."""

        for client in self.clients:
            client.abort_training()
        self.is_aborting = True

    def save_checkpoint(self, output_file_path: str) -> None:
        """Saves the current state of the global model to a checkpoint file.

        Args:
            output_path: The path to the file into which the model is to be saved.
        """

        torch.save(self.global_model.state_dict(), output_file_path)
