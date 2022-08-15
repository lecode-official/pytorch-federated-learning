"""Contains the federated-averaging command."""

import sys
import signal
import logging
from argparse import Namespace

import torch

from fl.commands.base import BaseCommand
from fl.datasets import create_dataset, split_dataset
from fl.federated_learning import FederatedLearningCentralServer, FederatedLearningClient


class FederatedAveragingCommand(BaseCommand):
    """Represents the federated-averaging command, which performs federated learning using federated averaging (FedAvg)."""

    def __init__(self) -> None:
        """Initializes a new FederatedAveragingCommand instance."""

        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)

        self.central_server = None
        self.is_aborting = False

    def run(self, command_line_arguments: Namespace) -> None:
        """Runs the command.

        Args:
            command_line_arguments (Namespace): The parsed command line arguments.
        """

        # Validates the command line arguments
        if command_line_arguments.dataset_path is None:
            self.logger.error('No dataset path was specified. Exiting.')
            sys.exit(1)
        if command_line_arguments.model_output_file_path is None:
            self.logger.warn('No output path was specified, so the trained global model will not be saved.')
        if command_line_arguments.number_of_clients > 250 and command_line_arguments.training_statistics_plot_output_file_path is not None:
            self.logger.warn('Plotting the training statistics plot for more than 250 clients will take a long time and is discouraged.')
        if command_line_arguments.number_of_clients > 1000 and command_line_arguments.training_statistics_plot_output_file_path is not None:
            self.logger.error('Plotting the training statistics plot for more than 1000 will take too long. Existing.')
            sys.exit(1)

        # Selects the device the training and validation will be performed on
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if command_line_arguments.cpu:
            device = 'cpu'

        # Loading the datasets
        self.logger.info('Loading dataset (%s)...', command_line_arguments.dataset)
        training_subset, validation_subset, sample_shape, number_of_classes = create_dataset(
            command_line_arguments.dataset,
            command_line_arguments.dataset_path
        )
        client_subsets = split_dataset(training_subset, command_line_arguments.number_of_clients)

        # Creates the clients
        clients = []
        self.logger.info('Creating %d clients...', command_line_arguments.number_of_clients)
        for index in range(command_line_arguments.number_of_clients):
            clients.append(FederatedLearningClient(
                index + 1,
                device,
                command_line_arguments.model,
                client_subsets[index],
                sample_shape,
                number_of_classes,
                command_line_arguments.learning_rate,
                command_line_arguments.momentum,
                command_line_arguments.weight_decay,
                command_line_arguments.batch_size
            ))

        # Creates the central server
        self.logger.info('Creating central server...')
        self.central_server = FederatedLearningCentralServer(
            clients,
            command_line_arguments.number_of_clients_per_communication_round,
            device,
            command_line_arguments.model,
            validation_subset,
            sample_shape,
            number_of_classes,
            command_line_arguments.batch_size
        )

        # Registers a signal handler, which graciously stops the training and saves the current state to disk, when the user hits Ctrl+C
        signal.signal(signal.SIGINT, lambda _, __: self.abort_training())

        # Performs the federated training for the specified number of communication rounds
        for communication_round in range(1, command_line_arguments.number_of_communication_rounds + 1):
            if self.is_aborting:
                self.logger.info('Graciously shutting down federated learning... Hit Ctrl+C again to force quit...')
                break
            self.logger.info('Starting communication round %d...', communication_round)
            self.central_server.train_clients_and_update_global_model(command_line_arguments.number_of_local_epochs)
            validation_loss, validation_accuracy = self.central_server.validate()
            self.logger.info(
                'Finished communication round %d, validation loss: %f, validation accuracy: %f',
                communication_round,
                validation_loss,
                validation_accuracy
            )

        # Saves the trained global to disk
        if command_line_arguments.model_output_file_path is not None:
            self.logger.info(
                'Finished federated training, saving trained global model to disk (%s)...',
                command_line_arguments.model_output_file_path
            )
            self.central_server.save_checkpoint(command_line_arguments.model_output_file_path)

        # Saves the training statistics plot
        if command_line_arguments.training_statistics_plot_output_file_path is not None:
            self.logger.info(
                'Plotting training statistics and saving the plot to disk (%s)...',
                command_line_arguments.training_statistics_plot_output_file_path
            )
            self.central_server.save_training_statistics_plot(command_line_arguments.training_statistics_plot_output_file_path)

    def abort_training(self) -> None:
        """Graciously aborts the federated learning."""

        # If the user hits Ctrl+C a second time, then the application is closed right away
        if self.is_aborting:
            exit()

        # Since this is the first time, that the user hit Ctrl+C, the aborting process is initiated
        self.is_aborting = True
        self.central_server.abort_training()
