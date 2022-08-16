"""Contains the federated-averaging command."""

import os
import csv
import signal
import logging
from datetime import datetime
from argparse import Namespace

import yaml
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

        # Makes sure that the output directory exists
        os.makedirs(command_line_arguments.output_path, exist_ok=True)

        # Prepares the training statistics CSV files by writing the headers to files
        with open(os.path.join(command_line_arguments.output_path, 'central-server-training-statistics.csv'), 'w') as training_statistics_file:
            csv_writer = csv.writer(training_statistics_file)
            csv_writer.writerow(['timestamp', 'communication_round', 'validation_loss', 'validation_accuracy'])
        with open(os.path.join(command_line_arguments.output_path, 'client-training-statistics.csv'), 'w') as training_statistics_file:
            csv_writer = csv.writer(training_statistics_file)
            csv_row = ['timestamp', 'communication_round']
            for client_id in range(1, command_line_arguments.number_of_clients + 1):
                csv_row.extend([f'client_{client_id}_training_loss', f'client_{client_id}_training_accuracy'])
            csv_writer.writerow(csv_row)

        # Saves the hyperparameters for later reference
        with open(os.path.join(command_line_arguments.output_path, 'hyperparameters.yaml'), 'w') as hyperparameters_file:
            yaml.dump({
                'method': 'federated_averaging',
                'number_of_clients': command_line_arguments.number_of_clients,
                'number_of_clients_per_communication_round': \
                    command_line_arguments.number_of_clients_per_communication_round or command_line_arguments.number_of_clients,
                'model': command_line_arguments.model_type,
                'dataset': command_line_arguments.dataset_type,
                'dataset_path': command_line_arguments.dataset_path,
                'number_of_communication_rounds': command_line_arguments.number_of_communication_rounds,
                'number_of_local_epochs': command_line_arguments.number_of_local_epochs,
                'output_path': command_line_arguments.output_path,
                'number_of_checkpoint_files_to_retain': command_line_arguments.number_of_checkpoint_files_to_retain,
                'learning_rate': command_line_arguments.learning_rate,
                'momentum': command_line_arguments.momentum,
                'weight_decay': command_line_arguments.weight_decay,
                'batch_size': command_line_arguments.batch_size,
                'force_cpu': command_line_arguments.force_cpu
            }, hyperparameters_file)

        # Selects the device the training and validation will be performed on
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if command_line_arguments.force_cpu:
            device = 'cpu'

        # Loading the datasets
        self.logger.info('Loading dataset (%s)...', command_line_arguments.dataset_type)
        training_subset, validation_subset, sample_shape, number_of_classes = create_dataset(
            command_line_arguments.dataset_type,
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
                command_line_arguments.model_type,
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
            command_line_arguments.model_type,
            validation_subset,
            sample_shape,
            number_of_classes,
            command_line_arguments.batch_size
        )

        # Registers a signal handler, which graciously stops the training and saves the current state to disk, when the user hits Ctrl+C
        signal.signal(signal.SIGINT, lambda _, __: self.abort_training())

        # Performs the federated training for the specified number of communication rounds
        current_greatest_validation_accuracy = 0
        retained_global_model_checkpoint_file_paths = []
        for communication_round in range(1, command_line_arguments.number_of_communication_rounds + 1):

            # If the user hit Ctrl+C, then the training is aborted
            if self.is_aborting:
                self.logger.info('Graciously shutting down federated learning... Hit Ctrl+C again to force quit...')
                break

            # Performs a communication round, where the clients (or in the case of client sub-sampling: a subset of the clients) is send the global
            # model and instructed to train the model on their local data, after that, the clients send back their updated models and the central
            # server aggregates them to form a new global model
            self.logger.info('Starting communication round %d...', communication_round)
            client_training_losses, client_training_accuracies = self.central_server.train_clients_and_update_global_model(
                command_line_arguments.number_of_local_epochs
            )

            # Validates the updated global model and reports its loss and accuracy
            central_server_validation_loss, central_server_validation_accuracy = self.central_server.validate()
            self.logger.info(
                'Finished communication round %d, validation loss: %f, validation accuracy: %f',
                communication_round,
                central_server_validation_loss,
                central_server_validation_accuracy * 100
            )

            # Writes the training statistics into CSV files (the training statistics of the central server and the clients are stored in a separate
            # CSV files)
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open(os.path.join(command_line_arguments.output_path, 'central-server-training-statistics.csv'), 'a') as training_statistics_file:
                csv_writer = csv.writer(training_statistics_file)
                csv_writer.writerow([timestamp, communication_round, central_server_validation_loss, central_server_validation_accuracy])
            with open(os.path.join(command_line_arguments.output_path, 'client-training-statistics.csv'), 'a') as training_statistics_file:
                csv_writer = csv.writer(training_statistics_file)
                csv_row = [timestamp, communication_round]
                for client_index in range(command_line_arguments.number_of_clients):
                    csv_row.extend([client_training_losses[client_index], client_training_accuracies[client_index]])
                csv_writer.writerow(csv_row)

            # If the updated global model has a better accuracy than any of its predecessors, then a checkpoint is saved for it, if the number of
            # checkpoint files that have already been saved, exceeds the number of checkpoint files to retain, then the oldest one is deleted (the
            # checkpoint file is not saved, if this is the last communication round, because the final model is saved anyway)
            if communication_round != command_line_arguments.number_of_communication_rounds:
                if central_server_validation_accuracy > current_greatest_validation_accuracy:

                    # Since the updated global model outperformed all previous global models, a checkpoint is saved for it
                    global_model_checkpoint_file_path = self.save_global_model_checkpoint(
                        command_line_arguments.model_type,
                        command_line_arguments.dataset_type,
                        communication_round,
                        central_server_validation_accuracy * 100,
                        command_line_arguments.output_path
                    )
                    current_greatest_validation_accuracy = central_server_validation_accuracy

                    # If the number of saved checkpoint files exceeds the number of checkpoint files that should be retained, the oldest checkpoint
                    # file is deleted
                    retained_global_model_checkpoint_file_paths.append(global_model_checkpoint_file_path)
                    if len(retained_global_model_checkpoint_file_paths) > command_line_arguments.number_of_checkpoint_files_to_retain:
                        global_model_checkpoint_file_to_remove = retained_global_model_checkpoint_file_paths[0]
                        os.remove(global_model_checkpoint_file_to_remove)
                        retained_global_model_checkpoint_file_paths = retained_global_model_checkpoint_file_paths[1:]

        # Saves the trained global model and the training statistics plot to disk
        self.logger.info('Finished federated training...')
        self.save_global_model_checkpoint(
            command_line_arguments.model_type,
            command_line_arguments.dataset_type,
            communication_round,
            central_server_validation_accuracy * 100,
            command_line_arguments.output_path
        )
        self.save_training_statistics_plot(command_line_arguments.output_path)

    def save_global_model_checkpoint(self, model_type: str, dataset_type: str, communication_round: str, accuracy: float, output_path: str) -> str:
        """Saves the current state of the global model of the central server to disk.

        Args:
            model_type (str): The type of model that is being trained.
            dataset_type (str): The type of dataset that the model being trained on.
            communication_round (str): The current communication round.
            accuracy (float): The accuracy of the model.
            output_path (str): The path to the directory into which the model checkpoint file is to be saved.

        Returns:
            str: Returns the path to the checkpoint file.
        """

        global_model_checkpoint_file_path = os.path.join(
            output_path,
            f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}-{model_type}-{dataset_type}-fedavg-{communication_round}-communication-round-'
            f'{accuracy:.2f}-accuracy.pt'
        )
        self.logger.info('Saving global model checkpoint to disk (%s)...', global_model_checkpoint_file_path)
        self.central_server.save_checkpoint(global_model_checkpoint_file_path)
        return global_model_checkpoint_file_path

    def save_training_statistics_plot(self, output_path: str) -> None:
        """Plots the training statistics and saves the resulting plot to disk.

        Args:
            output_path (str): The path to the directory into which the plot is to be saved.
        """

        training_statistics_plot_file_path = os.path.join(output_path, 'training-statistics-plot.png')
        self.logger.info('Plotting training statistics and saving the plot to disk (%s)...', training_statistics_plot_file_path)
        self.central_server.save_training_statistics_plot(training_statistics_plot_file_path)

    def abort_training(self) -> None:
        """Graciously aborts the federated learning."""

        # If the user hits Ctrl+C a second time, then the application is closed right away
        if self.is_aborting:
            exit()

        # Since this is the first time, that the user hit Ctrl+C, the aborting process is initiated
        self.is_aborting = True
        self.central_server.abort_training()
