"""Contains the base classes for commands and their descriptors."""

from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace


class BaseCommand(ABC):
    """Represents the base class for all commands in the application."""

    @abstractmethod
    def run(self, command_line_arguments: Namespace) -> None:
        """Runs the command.

        Args:
            command_line_arguments (Namespace): The parsed command line arguments.

        Raises:
            NotImplementedError: Since this is an abstract base class, NotImplementedError is raised.
        """

        raise NotImplementedError()


class BaseCommandDescriptor(ABC):
    """Represents a description of a command."""

    @abstractmethod
    def get_name(self) -> str:
        """Gets the name of the command.

        Returns:
            str: Returns the name of the command.

        Raises:
            NotImplementedError: Since this is an abstract base class, NotImplementedError is raised.
        """

        raise NotImplementedError()

    @abstractmethod
    def get_description(self) -> str:
        """Gets the description of the command.

        Returns:
            str: Returns the description of the command.

        Raises:
            NotImplementedError: Since this is an abstract base class, NotImplementedError is raised.
        """

        raise NotImplementedError()

    @abstractmethod
    def add_arguments(self, argument_parser: ArgumentParser) -> None:
        """Adds the command line arguments to the command line argument parser.

        Args:
            argument_parser (ArgumentParser): The command line argument parser to which the arguments are to be added.

        Raises:
            NotImplementedError: Since this is an abstract base class, NotImplementedError is raised.
        """

        raise NotImplementedError()
