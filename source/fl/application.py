"""Contains the actual Federated Learning Simulator application."""

__version__ = '0.1.0'

import sys
import logging
import argparse
import traceback

from fl.commands import get_command_descriptors, get_command


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

        self.commands = None

    def run(self) -> None:
        """Runs the application."""

        # Parses the command line arguments
        command_line_arguments = self.parse_command_line_arguments()

        # Finds the command that is to be run
        if command_line_arguments.command is None:
            self.logger.error('No command was specified.')
            sys.exit(1)
        try:
            command_class = get_command(command_line_arguments.command)
            command = command_class()
            command.run(command_line_arguments)
        except Exception:  # pylint: disable=broad-except
            self.logger.error('An error occurred in the command "%s": %s', command_line_arguments.command, traceback.format_exc())

    def parse_command_line_arguments(self) -> argparse.Namespace:
        """Parses the command line arguments of the application.

        Returns:
            argparse.Namespace: Returns a namespace containing the parsed command line arguments.
        """

        # Creates a command line argument parser for the application
        argument_parser = argparse.ArgumentParser(
            prog='fl',
            description='An implementation of federated learning using federated averaging (FedAvg).',
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

        # Adds the commands
        command_descriptors = get_command_descriptors()
        sub_parsers = argument_parser.add_subparsers(dest='command')
        for command_descriptor in command_descriptors:
            command_parser = sub_parsers.add_parser(command_descriptor.get_name(), help=command_descriptor.get_description(), add_help=False)
            command_parser.add_argument(
                '-h',
                '--help',
                action='help',
                help='Shows this help message and exits.'
            )
            command_descriptor.add_arguments(command_parser)

        # Parses the command line arguments and returns them
        return argument_parser.parse_args()
