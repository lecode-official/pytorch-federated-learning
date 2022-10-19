"""Models used in the federated learning."""

from enum import Enum
from typing import Type

import numpy
import torch


class ModelType(Enum):
    """Represents an enumeration of all the available model types."""

    LENET_5 = 'lenet-5'
    VGG11 = 'vgg11'

    @classmethod
    def available_models(cls: Type['ModelType']) -> list[str]:
        """Retrieves a list of the values for the available models.

        Returns:
            list[str]: Returns a list that contains the values for all available models.
        """

        return [model_type.value for model_type in cls]

    @classmethod
    def default_model(cls: Type['ModelType']) -> str:
        """Retrieves the value for the default model.

        Returns:
            str: Returns the value of the default model.
        """

        return ModelType.LENET_5.value

    def get_human_readable_name(self) -> str:
        """Retrieves a human-readable name for the model type.

        Returns:
            str: Returns a human-readable name for the model type.
        """

        human_readable_model_names_map = {
            ModelType.LENET_5: 'LeNet-5',
            ModelType.VGG11: 'VGG11'
        }
        return human_readable_model_names_map[self]


class NormalizationLayerKind(Enum):
    """Represents an enumeration of all the different normalization layer kinds that can be used."""

    BATCH_NORMALIZATION = 'batch-normalization'
    GROUP_NORMALIZATION = 'group-normalization'


class LeNet5(torch.nn.Module):
    """Represents the classical CNN model architecture LeNet5."""

    def __init__(self, input_shape: tuple[int, int, int], number_of_classes: int) -> None:
        """Initializes a new LeNet5 instance.

        Args:
            input_shape (tuple[int, int, int]): The shape of the data that is fed to the model as input.
            number_of_classes (int): The number of classes between which the model has to differentiate.
        """

        super().__init__()

        self.input_shape = input_shape
        self.number_of_classes = number_of_classes

        self.convolutional_layer_1 = torch.nn.Conv2d(in_channels=self.input_shape[0], out_channels=6, kernel_size=5)
        output_shape = (6, (self.input_shape[1] - 4) // 2, (self.input_shape[2] - 4) // 2)

        self.convolutional_layer_2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        output_shape = (16, (output_shape[1] - 4) // 2, (output_shape[2] - 4) // 2)

        input_size_fully_connected_layer_1 = numpy.prod(output_shape).item()
        self.fully_connected_layer_1 = torch.nn.Linear(in_features=input_size_fully_connected_layer_1, out_features=120)
        self.fully_connected_layer_2 = torch.nn.Linear(in_features=120, out_features=84)
        self.fully_connected_layer_3 = torch.nn.Linear(in_features=84, out_features=self.number_of_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass through the model.

        Args:
            x (torch.Tensor): The input features.

        Returns:
            torch.Tensor: Returns the outputs of the model.
        """

        y = self.convolutional_layer_1(x)
        y = torch.nn.functional.relu(y)
        y = torch.nn.functional.max_pool2d(y, kernel_size=2)

        y = self.convolutional_layer_2(y)
        y = torch.nn.functional.relu(y)
        y = torch.nn.functional.max_pool2d(y, kernel_size=2)

        y = y.flatten(start_dim=1)

        y = self.fully_connected_layer_1(y)
        y = torch.nn.functional.relu(y)

        y = self.fully_connected_layer_2(y)
        y = torch.nn.functional.relu(y)

        y = self.fully_connected_layer_3(y)
        y = torch.nn.functional.relu(y)

        return y


class Vgg11(torch.nn.Module):
    """Represents a VGG11, which is the smallest neural network of the VGG architecture family."""

    def __init__(self, input_shape: tuple[int, int, int], number_of_classes: int, normalization_layer_kind: NormalizationLayerKind) -> None:
        """Initializes a new Vgg11 instance.

        Args:
            input_shape (tuple[int, int, int]): The shape of the data that is fed to the model as input.
            number_of_classes (int): The number of classes between which the model has to differentiate.
            normalization_layer_kind (NormalizationLayerKind): The kind of the normalization layer that is used.
        """

        super().__init__()

        self.input_shape = input_shape
        self.number_of_classes = number_of_classes

        self.convolutional_layer_1 = torch.nn.Conv2d(in_channels=input_shape[0], out_channels=64, kernel_size=3, padding=1)
        if normalization_layer_kind == NormalizationLayerKind.BATCH_NORMALIZATION:
            self.normalization_layer_1 = torch.nn.BatchNorm2d(64)
        else:
            self.normalization_layer_1 = torch.nn.GroupNorm(num_groups=32, num_channels=64)
        output_shape = (64, self.input_shape[1] // 2, self.input_shape[2] // 2)

        self.convolutional_layer_2 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        if normalization_layer_kind == NormalizationLayerKind.BATCH_NORMALIZATION:
            self.normalization_layer_2 = torch.nn.BatchNorm2d(128)
        else:
            self.normalization_layer_2 = torch.nn.GroupNorm(num_groups=32, num_channels=128)
        output_shape = (128, output_shape[1] // 2, output_shape[2] // 2)

        self.convolutional_layer_3 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        if normalization_layer_kind == NormalizationLayerKind.BATCH_NORMALIZATION:
            self.normalization_layer_3 = torch.nn.BatchNorm2d(256)
        else:
            self.normalization_layer_3 = torch.nn.GroupNorm(num_groups=32, num_channels=256)

        self.convolutional_layer_4 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        if normalization_layer_kind == NormalizationLayerKind.BATCH_NORMALIZATION:
            self.normalization_layer_4 = torch.nn.BatchNorm2d(256)
        else:
            self.normalization_layer_4 = torch.nn.GroupNorm(num_groups=32, num_channels=256)
        output_shape = (256, output_shape[1] // 2, output_shape[2] // 2)

        self.convolutional_layer_5 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        if normalization_layer_kind == NormalizationLayerKind.BATCH_NORMALIZATION:
            self.normalization_layer_5 = torch.nn.BatchNorm2d(512)
        else:
            self.normalization_layer_5 = torch.nn.GroupNorm(num_groups=32, num_channels=512)

        self.convolutional_layer_6 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        if normalization_layer_kind == NormalizationLayerKind.BATCH_NORMALIZATION:
            self.normalization_layer_6 = torch.nn.BatchNorm2d(512)
        else:
            self.normalization_layer_6 = torch.nn.GroupNorm(num_groups=32, num_channels=512)
        output_shape = (512, output_shape[1] // 2, output_shape[2] // 2)

        self.convolutional_layer_7 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        if normalization_layer_kind == NormalizationLayerKind.BATCH_NORMALIZATION:
            self.normalization_layer_7 = torch.nn.BatchNorm2d(512)
        else:
            self.normalization_layer_7 = torch.nn.GroupNorm(num_groups=32, num_channels=512)

        self.convolutional_layer_8 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        if normalization_layer_kind == NormalizationLayerKind.BATCH_NORMALIZATION:
            self.normalization_layer_8 = torch.nn.BatchNorm2d(512)
        else:
            self.normalization_layer_8 = torch.nn.GroupNorm(num_groups=32, num_channels=512)
        output_shape = (512, output_shape[1] // 2, output_shape[2] // 2)

        self.fully_connected_layer_1 = torch.nn.Linear(512 * output_shape[1] * output_shape[2], 4096)
        self.fully_connected_layer_2 = torch.nn.Linear(4096, 4096)
        self.fully_connected_layer_3 = torch.nn.Linear(4096, number_of_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass through the model.

        Args:
            x (torch.Tensor): The input features.

        Returns:
            torch.Tensor: Returns the outputs of the model.
        """

        y = self.convolutional_layer_1(x)
        y = self.normalization_layer_1(y)
        y = torch.nn.functional.relu(y)
        y = torch.nn.functional.max_pool2d(y, kernel_size=2, stride=2)

        y = self.convolutional_layer_2(y)
        y = self.normalization_layer_2(y)
        y = torch.nn.functional.relu(y)
        y = torch.nn.functional.max_pool2d(y, kernel_size=2, stride=2)

        y = self.convolutional_layer_3(y)
        y = self.normalization_layer_3(y)
        y = torch.nn.functional.relu(y)

        y = self.convolutional_layer_4(y)
        y = self.normalization_layer_4(y)
        y = torch.nn.functional.relu(y)
        y = torch.nn.functional.max_pool2d(y, kernel_size=2, stride=2)

        y = self.convolutional_layer_5(y)
        y = self.normalization_layer_5(y)
        y = torch.nn.functional.relu(y)

        y = self.convolutional_layer_6(y)
        y = self.normalization_layer_6(y)
        y = torch.nn.functional.relu(y)
        y = torch.nn.functional.max_pool2d(y, kernel_size=2, stride=2)

        y = self.convolutional_layer_7(y)
        y = self.normalization_layer_7(y)
        y = torch.nn.functional.relu(y)

        y = self.convolutional_layer_8(y)
        y = self.normalization_layer_8(y)
        y = torch.nn.functional.relu(y)
        y = torch.nn.functional.max_pool2d(y, kernel_size=2, stride=2)

        y = torch.flatten(y, 1)  # pylint: disable=no-member

        y = self.fully_connected_layer_1(y)
        y = torch.nn.functional.relu(y)
        y = torch.nn.functional.dropout(y, 0.5)

        y = self.fully_connected_layer_2(y)
        y = torch.nn.functional.relu(y)
        y = torch.nn.functional.dropout(y, 0.5)

        y = self.fully_connected_layer_3(y)

        return y


def get_minimum_input_size(model_type: ModelType) -> tuple[int, int]:
    """Determines the minimum size of the input of the specified model.

    Args:
        model_type (ModelType): The type of model for which the minimum input size is to be determined.

    Raises:
        ValueError:
            If the specified model type is not supported, an exception is raised.

    Returns:
        tuple[int, int]: Returns a tuple containing the minimum supported height and the minimum supported width of the specified model.
    """

    if model_type == ModelType.LENET_5:
        return (16, 16)

    if model_type == ModelType.VGG11:
        return (32, 32)

    raise ValueError(f'The model type "{model_type.value}" is not supported.')


def create_model(
        model_type: ModelType,
        input_shape: tuple[int, int, int],
        number_of_classes: int,
        normalization_layer_kind: NormalizationLayerKind) -> torch.nn.Module:
    """Creates a model of the specified type.

    Args:
        model_type (ModelType): The type of model that is to be created.
        input_shape (tuple[int, int, int]): The shape of the data that is fed to the model as input.
        number_of_classes (int): The number of classes between which the model has to differentiate.
        normalization_layer_kind (NormalizationLayerKind): The kind of the normalization layer that is used.

    Raises:
        ValueError:
            If the specified model type is not supported, an exception is raised.

    Returns:
        torch.nn.Module: Returns the created model.
    """

    if model_type == ModelType.LENET_5:
        return LeNet5(input_shape, number_of_classes)

    if model_type == ModelType.VGG11:
        return Vgg11(input_shape, number_of_classes, normalization_layer_kind)

    raise ValueError(f'The model type "{model_type.value}" is not supported.')


def get_number_of_parameters(model: torch.nn.Module) -> int:
    """Retrieves the total number of trainable parameters of the specified model.

    Args:
        model (torch.nn.Module): The model for which the number of trainable parameters is to be retrieved.

    Returns:
        int: Returns the number of trainable parameters of the specified model.
    """

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
