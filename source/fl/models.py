"""Models used in the federated learning."""

import numpy
import torch


class LeNet5(torch.nn.Module):
    """Represents the classical CNN model architecture LeNet5."""

    def __init__(self, input_shape: tuple[int, int, int]) -> None:
        """Initializes a new LeNet5 instance."""

        super(LeNet5, self).__init__()

        self.input_shape = input_shape

        self.convolutional_layer_1 = torch.nn.Conv2d(in_channels=self.input_shape[0], out_channels=6, kernel_size=5)
        output_size_after_convolutional_layer_1 = (6, self.input_shape[1] - 4, self.input_shape[2] - 4)
        output_size_after_max_pooling_1 = (6, output_size_after_convolutional_layer_1[1] // 2, output_size_after_convolutional_layer_1[2] // 2)
        self.convolutional_layer_2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        output_size_after_convolutional_layer_2 = (16, output_size_after_max_pooling_1[1] - 4, output_size_after_max_pooling_1[2] - 4)
        output_size_after_max_pooling_2 = (16, output_size_after_convolutional_layer_2[1] // 2, output_size_after_convolutional_layer_2[2] // 2)

        input_size_fully_connected_layer_1 = numpy.prod(output_size_after_max_pooling_2).item()
        self.fully_connected_layer_1 = torch.nn.Linear(in_features=input_size_fully_connected_layer_1, out_features=120)
        self.fully_connected_layer_2 = torch.nn.Linear(in_features=120, out_features=84)
        self.fully_connected_layer_3 = torch.nn.Linear(in_features=84, out_features=10)

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
