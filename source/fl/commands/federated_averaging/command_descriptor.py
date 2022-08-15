"""Contains the descriptor for the federated-averaging command."""

from argparse import ArgumentParser

import fl.models
import fl.datasets
from fl.commands.base import BaseCommandDescriptor


class FederatedAveragingCommand(BaseCommandDescriptor):
    """Represents the descriptor for the federated-averaging command."""

    def get_name(self) -> str:
        """Gets the name of the command.

        Returns:
            str: Returns the name of the command.
        """

        return 'federated-averaging'

    def get_description(self) -> str:
        """Gets the description of the command.

        Returns:
            str: Returns the description of the command.
        """

        return 'Performs federating learning using federated averaging (FedAvg).'

    def add_arguments(self, argument_parser: ArgumentParser) -> None:
        """Adds the command line arguments to the command line argument parser.

        Args:
            argument_parser (ArgumentParser): The command line argument parser to which the arguments are to be added.
        """

        argument_parser.add_argument(
            '-n',
            '--number-of-clients',
            dest='number_of_clients',
            type=int,
            default=10,
            help='The number of federated learning clients. Defaults to 10.'
        )
        argument_parser.add_argument(
            '-N',
            '--number-of-clients-per-communication-round',
            dest='number_of_clients_per_communication_round',
            type=int,
            default=None,
            help='''One of the primary bottlenecks in the communication between the central server and its clients is the number of clients that the
                central server has to communicate with in each communication round. One easy method of reducing this overhead, is to subsample the
                client population. In each communication round, the central server only selects a subset of clients, which will train and communicate
                their updates back. This parameter specifies the number of clients that will be selected at random in each communication round.
                Defaults to the number of clients.
            '''
        )
        argument_parser.add_argument(
            '-m',
            '--model',
            dest='model',
            type=str,
            choices=fl.models.AVAILABLE_MODELS,
            default=fl.models.DEFAULT_MODEL,
            help='The model that is to be used for the training. Defaults to "lenet-5".'
        )
        argument_parser.add_argument(
            '-d',
            '--dataset',
            dest='dataset',
            type=str,
            choices=fl.datasets.AVAILABLE_DATASETS,
            default=fl.datasets.DEFAULT_DATASET,
            help='The dataset that is to be used for the training. Defaults to "mnist".'
        )
        argument_parser.add_argument(
            '-D',
            '--dataset-path',
            dest='dataset_path',
            type=str,
            required=True,
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
            default=5,
            help='The number of communication epochs for which the clients are training the model on their local data. Defaults to 5.'
        )
        argument_parser.add_argument(
            '-o',
            '--output-path',
            dest='output_path',
            type=str,
            required=True,
            help='''The path to the directory into which checkpoints of the trained global model as well as training statistics are to be stored.
                If the directory does not exist, it is created.
            '''
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
            default=16,
            help='The size of mini-batches that are to be used during training and validation. Defaults to 128.'
        )
        argument_parser.add_argument(
            '-c',
            '--cpu',
            dest='cpu',
            action='store_true',
            help='Always use the CPU for training, even when a GPU is available.'
        )
