"""Models used in the federated learning."""

import numpy
import torch


AVAILABLE_MODELS = ['lenet-5']
DEFAULT_MODEL = 'lenet-5'


class LeNet5(torch.nn.Module):
    """Represents the classical CNN model architecture LeNet5."""

    def __init__(self, input_shape: tuple[int, int, int], number_of_classes: int) -> None:
        """Initializes a new LeNet5 instance.

        Args:
            input_shape (tuple[int, int, int]): The shape of the data that is fed to the model as input.
            number_of_classes (int): The number of classes between which the model has to differentiate.
        """

        super(LeNet5, self).__init__()

        self.input_shape = input_shape
        self.number_of_classes = number_of_classes

        self.convolutional_layer_1 = torch.nn.Conv2d(in_channels=self.input_shape[0], out_channels=6, kernel_size=5)
        output_size_after_convolutional_layer_1 = (6, self.input_shape[1] - 4, self.input_shape[2] - 4)
        output_size_after_max_pooling_1 = (6, output_size_after_convolutional_layer_1[1] // 2, output_size_after_convolutional_layer_1[2] // 2)
        self.convolutional_layer_2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        output_size_after_convolutional_layer_2 = (16, output_size_after_max_pooling_1[1] - 4, output_size_after_max_pooling_1[2] - 4)
        output_size_after_max_pooling_2 = (16, output_size_after_convolutional_layer_2[1] // 2, output_size_after_convolutional_layer_2[2] // 2)

        input_size_fully_connected_layer_1 = numpy.prod(output_size_after_max_pooling_2).item()
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


def create_model(model_type: str, input_shape: tuple[int, int, int], number_of_classes: int) -> torch.nn.Module:
    """Creates a model of the specified type.

    Args:
        model_type (str): The type of model that is to be created.
        input_shape (tuple[int, int, int]): The shape of the data that is fed to the model as input.
        number_of_classes (int): The number of classes between which the model has to differentiate.

    Raises:
        ValueError:
            If the specified model type is not supported, an exception is raised.

    Returns:
        torch.nn.Module: Returns the created model.
    """

    if model_type == 'lenet-5':
        return LeNet5(input_shape, number_of_classes)

    raise ValueError(f'The model type "{model_type}" is not supported.')
