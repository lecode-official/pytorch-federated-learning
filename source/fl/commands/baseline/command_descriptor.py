"""Contains the descriptor for the baseline command."""

from argparse import ArgumentParser

import fl.models
import fl.datasets
from fl.commands.base import BaseCommandDescriptor


class BaselineCommandDescriptor(BaseCommandDescriptor):
    """Represents the descriptor for the baseline command."""

    def get_name(self) -> str:
        """Gets the name of the command.

        Returns:
            str: Returns the name of the command.
        """

        return 'baseline'

    def get_description(self) -> str:
        """Gets the description of the command.

        Returns:
            str: Returns the description of the command.
        """

        return 'Performs non-federating learning as a baseline to compare federated learning algorithms against.'

    def add_arguments(self, argument_parser: ArgumentParser) -> None:
        """Adds the command line arguments to the command line argument parser.

        Args:
            argument_parser (ArgumentParser): The command line argument parser to which the arguments are to be added.
        """

        argument_parser.add_argument(
            '-m',
            '--model',
            dest='model_type',
            type=str,
            choices=fl.models.AVAILABLE_MODELS,
            default=fl.models.DEFAULT_MODEL,
            help=f'The model that is to be used for the training. Defaults to "{fl.models.DEFAULT_MODEL}".'
        )
        argument_parser.add_argument(
            '-d',
            '--dataset',
            dest='dataset_type',
            type=str,
            choices=fl.datasets.AVAILABLE_DATASETS,
            default=fl.datasets.DEFAULT_DATASET,
            help=f'The dataset that is to be used for the training. Defaults to "{fl.datasets.DEFAULT_DATASET}".'
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
            '--number-of-epochs',
            dest='number_of_epochs',
            type=int,
            default=25,
            help='The number of epochs to train the model for. Defaults to 25.'
        )
        argument_parser.add_argument(
            '-o',
            '--output-path',
            dest='output_path',
            type=str,
            required=True,
            help='''The path to the directory into which checkpoints of the trained model as well as training statistics are to be stored. If the
                directory does not exist, it is created.
            '''
        )
        argument_parser.add_argument(
            '-R',
            '--number-of-checkpoint-files-to-retain',
            dest='number_of_checkpoint_files_to_retain',
            type=int,
            default=5,
            help='''After each epochs, the current state of the model is saved to a checkpoint file if its validation accuracy is greater than any
                previous state of the model. This can potentially result in a great number of checkpoint files that are being stored. This argument
                controls how many past checkpoint files are being retained. If the number of past checkpoint files exceeds this argument, then the
                oldest one is deleted. Defaults to 5.
            '''
        )
        argument_parser.add_argument(
            '-l',
            '--learning-rate',
            dest='learning_rate',
            type=float,
            default=0.01,
            help='''The initial learning rate of the optimizer. Depending on the --learning-rate-decay argument, the learning rate will be decayed
                during the training. Defaults to 0.01.
            '''
        )
        argument_parser.add_argument(
            '-L',
            '--learning-rate-decay',
            dest='learning_rate_decay',
            type=float,
            default=0.95,
            help='''The learning rate is decayed exponentially during the training. This argument is the decay rate of the learning rate. A decay rate
                1.0 would result in no decay at all. Defaults to 0.95.
            '''
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
            '--force-cpu',
            dest='force_cpu',
            action='store_true',
            help='Always use the CPU for training, even when a GPU is available.'
        )
