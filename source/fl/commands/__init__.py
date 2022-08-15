"""Contains the command line commands of the application."""

import os
import importlib
from glob import glob
from inspect import getmembers, isclass

from fl.commands.base import BaseCommand, BaseCommandDescriptor


def get_command_descriptors() -> list[BaseCommandDescriptor]:
    """Retrieves a list of descriptors for all the commands of the application.

    Returns:
        list[CommandDescriptor]: Returns a list containing the descriptors of all the commands of the application.
    """

    # Gets all command descriptor modules that are in the commands sub-package
    command_descriptor_modules = []
    for module_path in glob(os.path.join(os.path.dirname(__file__), '**', 'command_descriptor.py')):
        sub_package_path = os.path.split(module_path)[0]
        sub_package_name = os.path.split(sub_package_path)[1]
        command_descriptor_modules.append(importlib.import_module(f'fl.commands.{sub_package_name}.command_descriptor'))

    # Gets the command classes, which are all the classes in the commands module and its sub-modules that inherit from BaseCommand
    command_descriptors = []
    for command_descriptor_module in command_descriptor_modules:
        for _, command_descriptor_class in getmembers(command_descriptor_module, isclass):
            if BaseCommandDescriptor in command_descriptor_class.__bases__ and command_descriptor_class not in command_descriptors:
                command_descriptors.append(command_descriptor_class())

    # Returns the list of command descriptors
    return command_descriptors


def get_command(command_name: str) -> BaseCommand:
    """Retrieves the command with the specified name.

    Args:
        command_name (str): The name of the command that is to be retrieved.

    Returns:
        BaseCommand: Returns the command with the specified name.

    Raises:
        ValueError: If the command class for the command cannot be found, then a ValueError is raised.
    """

    # Loads to the module that contains the command, the name of the command is in kebab case (lowercase with dashes between words), so it is
    # converted to snake case (lowercase with underscores between words) to get the name of the module that contains the command
    command_module_name = command_name.lower().replace('-', '_')
    command_module = importlib.import_module(f'fl.commands.{command_module_name}.command')

    # Gets the class of the command, the name of the command class is in pascal case (the first character of a word is uppercase and no separators
    # between words, there are exceptions, that are unfortunately handled manually)
    lower_case_command_class_name = command_module_name.replace('_', '') + 'command'
    command_module_classes = getmembers(command_module, isclass)
    command_class = None
    for class_name, class_object in command_module_classes:
        if class_name.lower() == lower_case_command_class_name:
            command_class = class_object
    if command_class is None:
        raise ValueError(f'No command class for the command {command_name} found.')
    return command_class
