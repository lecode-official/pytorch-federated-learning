"""Contains the actual entrypoint to the application."""

__version__ = '0.1.0'

import sys
import logging
import argparse

import torch

from fl.datasets import create_dataset, split_dataset
from fl.federated_learning import FederatedLearningCentralServer, FederatedLearningClient


class Application:
    """Represents the federated learning application."""


    def __init__(self) -> None:
        """Initializes a new Application instance."""

        self.logger = logging.getLogger('fl')
        self.logger.setLevel(logging.DEBUG)
        logging_formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
        console_logging_handler = logging.StreamHandler(sys.stdout)
        console_logging_handler.setLevel(logging.DEBUG)
        console_logging_handler.setFormatter(logging_formatter)
        self.logger.addHandler(console_logging_handler)

    def run(self) -> None:
        """Runs the application."""

        # Parses the command line arguments
        command_line_arguments = self.parse_command_line_arguments()
        if command_line_arguments.dataset_path is None:
            self.logger.error('No dataset path was specified. Exiting.')
            sys.exit(1)
        if command_line_arguments.model_output_file_path is None:
            self.logger.warn('No output path was specified, so the trained global model will not be saved.')

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
        central_server = FederatedLearningCentralServer(
            clients,
            device,
            command_line_arguments.model,
            validation_subset,
            sample_shape,
            number_of_classes,
            command_line_arguments.batch_size
        )

        # Performs the federated training for the specified number of communication rounds
        for communication_round in range(1, command_line_arguments.number_of_communication_rounds + 1):
            self.logger.info('Starting communication round %d...', communication_round)
            central_server.train_clients_and_update_global_model(command_line_arguments.number_of_local_epochs)
            validation_loss, validation_accuracy = central_server.validate()
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
            central_server.save_checkpoint(command_line_arguments.model_output_file_path)

        # Saves the training statistics plot
        if command_line_arguments.statistics_plot_output_file_path is not None:
            self.logger.info(
                'Plotting training statistics and saving the plot to disk (%s)...',
                command_line_arguments.statistics_plot_output_file_path
            )
            central_server.save_statistics_plot(command_line_arguments.statistics_plot_output_file_path)

    def parse_command_line_arguments(self) -> argparse.Namespace:
        """Parses the command line arguments of the application.

        Returns:
            argparse.Namespace: Returns a namespace containing the parsed command line arguments.
        """

        # Creates a command line argument parser for the application
        argument_parser = argparse.ArgumentParser(
            prog='fl',
            description='An implementation of vanilla federated learning using federated averaging (FedAvg).',
            add_help=False
        )

        # Adds the command line argument that displays the help message
        argument_parser.add_argument(
            '-h',
            '--help',
            action='help',
            help='Shows this help message and exits.'
        )

        # Adds the command line argument for the version of the application
        argument_parser.add_argument(
            '-v',
            '--version',
            action='version',
            version=f'Federated Learning Simulator {__version__}',
            help='Displays the version string of the application and exits.'
        )

        # Adds the command line arguments for the federated learning
        argument_parser.add_argument(
            '-n',
            '--number-of-clients',
            dest='number_of_clients',
            type=int,
            default=10,
            help='The number of federated learning clients. Defaults to 10.'
        )
        argument_parser.add_argument(
            '-m',
            '--model',
            dest='model',
            type=str,
            choices=['lenet-5'],
            default='lenet-5',
            help='The model that is to be used for the training. Defaults to "lenet-5".'
        )
        argument_parser.add_argument(
            '-d',
            '--dataset',
            dest='dataset',
            type=str,
            choices=['cifar-10', 'mnist'],
            default='mnist',
            help='The dataset that is to be used for the training. Defaults to "mnist".'
        )
        argument_parser.add_argument(
            '-D',
            '--dataset-path',
            dest='dataset_path',
            type=str,
            help='''The path to the directory that contains the dataset that is to be used for the training. If the dataset does not exist, it is
                downloaded automatically.
            '''
        )
        argument_parser.add_argument(
            '-r',
            '--number-of-communication-rounds',
            dest='number_of_communication_rounds',
            type=int,
            default=50,
            help='''The number of communication rounds of the federated learning. One communication round consists of sending the global model to the
                clients, instructing them to perform training on their local dataset, and aggregating their updated local models into a new global
                model. Defaults to 50.
            '''
        )
        argument_parser.add_argument(
            '-e',
            '--number-of-local-epochs',
            dest='number_of_local_epochs',
            type=int,
            default=2,
            help='The number of communication epochs for which the clients are training the model on their local data. Defaults to 5.'
        )
        argument_parser.add_argument(
            '-o',
            '--model-output-file-path',
            dest='model_output_file_path',
            type=str,
            help='The path to the file into which the trained global model is to be stored. If no path is specified, the model is not saved.'
        )
        argument_parser.add_argument(
            '-p',
            '--statistics-plot-output-file-path',
            dest='statistics_plot_output_file_path',
            type=str,
            help='The path to the file into which the training statistics plot is to be stored. If no path is specified, the plot is not saved.'
        )
        argument_parser.add_argument(
            '-l',
            '--learning-rate',
            dest='learning_rate',
            type=float,
            default=0.01,
            help='The learning rate of the optimizer. Defaults to 0.01.'
        )
        argument_parser.add_argument(
            '-M',
            '--momentum',
            dest='momentum',
            type=float,
            default=0.9,
            help='The momentum of the optimizer. Defaults to 0.9.'
        )
        argument_parser.add_argument(
            '-w',
            '--weight-decay',
            dest='weight_decay',
            type=float,
            default=0.0005,
            help='The rate at which the weights are decayed during optimization. Defaults to 0.96.'
        )
        argument_parser.add_argument(
            '-b',
            '--batch-size',
            dest='batch_size',
            type=int,
            default=128,
            help='The size of mini-batches that are to be used during training and validation. Defaults to 128.'
        )
        argument_parser.add_argument(
            '-c',
            '--cpu',
            dest='cpu',
            action='store_true',
            help='Always use the CPU for training, even when a GPU is available.'
        )

        # Parses the command line arguments and returns them
        return argument_parser.parse_args()
