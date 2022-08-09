"""Models used in the federated learning."""

import torch


class LeNet5(torch.nn.Module):
    """Represents the classical CNN model architecture LeNet5."""

    def __init__(self):
        """Initializes a new LeNet5 instance."""

        super(LeNet5, self).__init__()

        self.convolutional_layer_1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.convolutional_layer_2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        self.fully_connected_layer_1 = torch.nn.Linear(in_features=256, out_features=120)
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
