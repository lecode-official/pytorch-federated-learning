"""Datasets used to train federated models."""

import torch
import torchvision


def load_cifar_10(path: str) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Loads the training and validation subsets of the CIFAR-10 dataset.

    Args:
        path (str): The path to the directory that contains the CIFAR-10 dataset. If the dataset does not exists at that location, then it is
            downloaded automatically.

    Returns:
        tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]: Returns a tuple containing the training and the validation subsets of the CIFAR-10
            dataset.
    """

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.1307], std=[0.3081])
    ])
    training_subset = torchvision.datasets.CIFAR10(root=path, train=True, download=True, transform=transform)
    validation_subset = torchvision.datasets.CIFAR10(root=path, train=False, download=True, transform=transform)

    return training_subset, validation_subset
