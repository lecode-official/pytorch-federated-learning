"""Implementation of the federated learning protocol and federated averaging (FedAvg)."""

import logging
import collections
from enum import Enum
from typing import Union

import torch
import matplotlib
from matplotlib import pyplot

from fl.models import create_model
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
            local_model_type: str,
            local_training_subset: torch.utils.data.Dataset,
            sample_shape: tuple,
            number_of_classes: int,
            learning_rate: float = 0.01,
            momentum: float = 0.9,
            weight_decay: float = 0.0005,
            batch_size: int = 128) -> None:
        """Initializes a new FederatedLearningClient instance.

        Args:
            device (Union[str, torch.device]): The device on which the local model is to be trained.
            local_model_type (str): The type of local model that is to be trained.
            local_training_subset (torch.utils.data.Dataset): The training subset of the local dataset on which the model is to be trained.
            sample_shape (tuple): The shape of the samples in the dataset.
            number_of_classes (int): The number of classes in the dataset.
            learning_rate (float, optional): The learning rate of the optimizer. Defaults to 0.01.
            momentum (float, optional): The momentum of the optimizer. Defaults to 0.9.
            weight_decay (float, optional): The rate at which the weights are decayed during optimization. Defaults to 0.0005.
            batch_size (int, optional): The size of mini-batches that are to be used during training. Defaults to 128.
        """

        self.device = device
        self.local_model_type = local_model_type
        self.local_training_subset = local_training_subset
        self.sample_shape = sample_shape
        self.number_of_classes = number_of_classes
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.batch_size = batch_size

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

        # Creates the local model
        local_model = create_model(self.local_model_type, input_shape=self.sample_shape, number_of_classes=self.number_of_classes)

        # Copies the parameters of the global model
        local_model_parameters = local_model.state_dict()
        for parameter_name in global_model_parameters:
            local_model_parameters[parameter_name].copy_(global_model_parameters[parameter_name])

        # Creates the trainer for the model
        trainer = Trainer(
            self.device,
            local_model,
            self.local_training_subset,
            self.learning_rate,
            self.momentum,
            self.weight_decay,
            self.batch_size
        )

        # Trains the local model for the specified amount of epochs and saves the training loss and accuracy
        training_loss = None
        training_accuracy = None
        for _ in range(number_of_epochs):
            training_loss, training_accuracy = trainer.train_for_one_epoch()

        # Returns the training loss, training accuracy, and the updated parameters of the local model
        return training_loss, training_accuracy, trainer.model.state_dict()


class FederatedLearningCentralServer:
    """Represents a federated learning central server, which coordinates the federated learning of the global model."""

    def __init__(
            self,
            clients: list[FederatedLearningClient],
            device: Union[str, torch.device],
            global_model_type: str,
            central_validation_subset: torch.utils.data.Dataset,
            sample_shape: tuple,
            number_of_classes: int,
            batch_size: int = 128) -> None:
        """Initializes a new FederatedLearningCentralServer instance.

        Args:
            clients (list[FederatedLearningClient]): The federated learning clients.
            device (Union[str, torch.device]): The device on which the global model of the central server is to be validated.
            global_model_type (str): The type of model that is to be used as global model for the central server.
            central_validation_subset (torch.utils.data.Dataset): The validation subset on which the global model is to be validated.
            sample_shape (tuple): The shape of the samples in the dataset.
            number_of_classes (int): The number of classes in the dataset.
            batch_size (int, optional): The size of the mini-batches that are to be used during the validation. Defaults to 128.
        """

        # Stores the arguments for later use
        self.clients = clients
        self.device = device
        self.global_model_type = global_model_type
        self.central_validation_subset = central_validation_subset
        self.sample_shape = sample_shape
        self.number_of_classes = number_of_classes
        self.batch_size = batch_size

        # Initializes the logger
        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)

        # Creates the global model and the validator for it
        self.global_model = create_model(self.global_model_type, input_shape=self.sample_shape, number_of_classes=self.number_of_classes)
        self.validator = Validator(self.device, self.global_model, self.central_validation_subset, self.batch_size)

        # Initializes the federated averaging algorithm
        self.model_aggregation_strategy = FederatedAveraging(self.global_model)

        # Initializes the statistics
        self.client_training_losses = [[] for _ in self.clients]
        self.client_training_accuracies = [[] for _ in self.clients]
        self.central_server_validation_losses = []
        self.central_server_validation_accuracies = []

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
            self.client_training_losses[index].append(training_loss)
            self.client_training_accuracies[index].append(training_accuracy)
            self.logger.info('Finished training client %d, Training loss: %f, training accuracy %f', index + 1, training_loss, training_accuracy)
            self.model_aggregation_strategy.add_local_model(local_model_parameters)

        # Updates the parameters of the global model by aggregating the updated parameters of the clients using federated averaging (FedAvg)
        self.model_aggregation_strategy.update_global_model(self.global_model)

    def validate(self) -> tuple[float, float]:
        """Validates the global model of the central server.

        Returns:
            tuple[float, float]: Returns the validation loss and the validation accuracy of the global model.
        """

        validation_loss, validation_accuracy = self.validator.validate()
        validation_loss = validation_loss
        validation_accuracy = validation_accuracy

        self.central_server_validation_losses.append(validation_loss)
        self.central_server_validation_accuracies.append(validation_accuracy)

        return validation_loss, validation_accuracy

    def save_statistics_plot(self, output_file_path: str) -> None:
        """Plots the training statistics and save the resulting plot to a file.

        Args:
            output_file_path (str): The path to the file into which the statistics plot is to be saved.
        """

        # Makes sure that Matplotlib uses a similar font to LaTeX, so that the figures are consistent with LaTeX documents
        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        matplotlib.rcParams['font.family'] = 'STIXGeneral'

        # Creates the figure
        width, height = self.determine_optimal_grid_size(len(self.clients), prefer_larger_width=False)
        figure = pyplot.figure(figsize=(10, 5), dpi=300, tight_layout=True)
        grid_specification = figure.add_gridspec(ncols=width+1, nrows=height, width_ratios=[2 * width] + [1] * width)

        # Determines the limits of the y-axis for the loss, so that the y-axes of the central server and the clients are all on the same scale (the
        # accuracy is bounded between 0 and 1, but the loss can grow almost arbitrarily)
        loss_axis_upper_y_limit = max([loss for losses in self.client_training_losses for loss in losses] + self.central_server_validation_losses)

        # Determines the final validation loss and validation accuracy of the global model
        final_central_server_validation_loss = self.central_server_validation_losses[-1]
        final_central_server_validation_accuracy = self.central_server_validation_accuracies[-1]

        # Creates the plot for the validation loss and validation accuracy of the central server
        communication_rounds = list(range(1, len(self.central_server_validation_accuracies) + 1))
        central_server_validation_accuracy_axis = figure.add_subplot(grid_specification[:, 0])
        central_server_validation_accuracy_axis.set_ylim(0.0, 1.0)
        central_server_validation_accuracy_axis.set_xlabel('Communication Rounds')
        central_server_validation_accuracy_axis.set_ylabel('Validation Accuracy')
        central_server_validation_accuracy_axis.set_title('Central Server')
        central_server_validation_accuracy_axis.plot(
            communication_rounds,
            self.central_server_validation_accuracies,
            color='blue',
            linewidth=0.5,
            label=f'Accuracy (Final Accuracy: {final_central_server_validation_accuracy:.2})'
        )
        accuracy_handles, accuracy_labels = central_server_validation_accuracy_axis.get_legend_handles_labels()
        central_server_validation_loss_axis = central_server_validation_accuracy_axis.twinx()
        central_server_validation_loss_axis.set_ylabel('Validation Loss')
        central_server_validation_loss_axis.plot(
            communication_rounds,
            self.central_server_validation_losses,
            color='red',
            linewidth=0.5,
            label=f'Loss (Final Loss: {final_central_server_validation_loss:.2})'
        )
        central_server_validation_loss_axis.set_ylim((0.0, loss_axis_upper_y_limit))
        loss_handles, loss_labels = central_server_validation_loss_axis.get_legend_handles_labels()
        central_server_validation_accuracy_axis.legend(accuracy_handles + loss_handles, accuracy_labels + loss_labels)

        # Creates the plots for the training loss and training accuracy of the clients
        federated_learning_client_index = 0
        for column in range(1, width + 1):
            for row in range(height):
                federated_learning_client_training_accuracy_axis = figure.add_subplot(grid_specification[row, column])
                federated_learning_client_training_accuracy_axis.tick_params(
                    color='white',
                    left=False,
                    bottom=False,
                    labelleft=False,
                    labelbottom=False
                )
                federated_learning_client_training_accuracy_axis.set_ylim(0.0, 1.0)
                federated_learning_client_training_accuracy_axis.text(
                    0.5,
                    0.1,
                    str(federated_learning_client_index + 1),
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize='small',
                    transform=federated_learning_client_training_accuracy_axis.transAxes
                )
                federated_learning_client_training_accuracy_axis.plot(
                    communication_rounds,
                    self.client_training_accuracies[federated_learning_client_index],
                    color='blue',
                    linewidth=0.5
                )
                federated_learning_client_training_loss_axis = federated_learning_client_training_accuracy_axis.twinx()
                federated_learning_client_training_loss_axis.set_ylim((0.0, loss_axis_upper_y_limit))
                federated_learning_client_training_loss_axis.tick_params(right=False, labelright=False)
                federated_learning_client_training_loss_axis.plot(
                    communication_rounds,
                    self.client_training_losses[federated_learning_client_index],
                    color='red',
                    linewidth=0.5
                )
                federated_learning_client_index += 1

        # Creates an invisible axis, which is just a hack to place a title above the client plots
        federated_learning_client_title_axis = figure.add_subplot(grid_specification[:, 1:])
        federated_learning_client_title_axis.set_xticks([])
        federated_learning_client_title_axis.set_yticks([])
        federated_learning_client_title_axis.spines['right'].set_visible(False)
        federated_learning_client_title_axis.spines['top'].set_visible(False)
        federated_learning_client_title_axis.spines['bottom'].set_visible(False)
        federated_learning_client_title_axis.spines['left'].set_visible(False)
        federated_learning_client_title_axis.set_facecolor('none')
        federated_learning_client_title_axis.set_title('Clients')

        # Saves the plot
        figure.savefig(output_file_path)

    def determine_optimal_grid_size(self, number_of_elements: int, prefer_larger_width: bool) -> list[int]:
        """Determines the optimal edge lengths for a grid that should contain the specified number of elements. Each number of elements has multiple
        grids in which they can be arranged, but the optimal grid size is the one where both sides are as large as possible.

        Args:
            number (int): The number of elements that should be contained in the grid.
            prefer_larger_width (bool): For element counts that are not perfect squares, there are always two optimal grid sizes: one where the width
                is larger and one where the height is larger. This parameter controls which of the two is selected. If True, the grid size with a
                larger width is selected and if False, the grid size with a larger height is selected. For example, if there are 50 elements, then the
                optimal grid sizes are 5 by 10 and 10 by 5 elements. If larger widths are preferred, then 10 by 5 elements will be chosen as the
                optimal grid size, otherwise 5 by 10 will be chosen.

        Returns:
            tuple[int, int]: Returns the optimal grid size as tuple, where the first element is the width and the second element is the height.
        """

        # Grid sizes are always composed of widths and heights that are integer divisors of the number of elements, therefore all divisors of the
        # number of elements are determined
        grid_sizes = []
        for divisor_candidate in range(1, number_of_elements + 1):
            if number_of_elements % divisor_candidate == 0:
                grid_sizes.append((divisor_candidate, number_of_elements // divisor_candidate))

        # To find the optimal grid size, the grid sizes are ordered by their sum, the optimal grid sizes are always the ones where the sum of width
        # and height are lowest, if the number of elements is not a perfect square then there are always two optimal grid sizes, one where the width
        # is larger and one where the height is larger, therefore the grid sizes are then ordered by their width or height, which is controlled by the
        # prefer_larger_width parameter (if larger widths should be preferred, then the grid sizes are ordered by height, because the grid sizes are
        # ordered in ascending order and the grid size with the lower height will be first, the same applies accordingly when larger heights should be
        # preferred, in which case the grid sizes are ordered by width)
        grid_sizes = sorted(grid_sizes, key=lambda grid_size: (sum(grid_size), grid_size[1 if prefer_larger_width else 0]))

        # Now the optimal grid size is the first element in the list of all grid sizes
        return grid_sizes[0]

    def save_checkpoint(self, output_file_path: str) -> None:
        """Saves the current state of the global model to a file.

        Args:
            output_path: The path to the file into which the model is to be saved.
        """

        torch.save(self.global_model.state_dict(), output_file_path)
