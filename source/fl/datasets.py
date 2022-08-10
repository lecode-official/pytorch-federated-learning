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


def split_dataset(dataset: torch.utils.data.Dataset, number_of_clients: int) -> list[torch.utils.data.Dataset]:
    """Splits the specified dataset evenly among the specified number of clients.

    Args:
        dataset (torch.utils.data.Dataset): The dataset that is to be split.
        number_of_clients (int): The number of clients among which the dataset is to be split.

    Returns:
        tuple[torch.utils.data.Dataset, list[torch.utils.data.Dataset]]: Returns a list that contains subsets of the specified dataset for every
            federated learning client.
    """

    # Determines the sizes of local datasets of the clients, since the number of samples in the dataset might not be divisible by the number of
    # clients, the first n-1 clients get an equal share of the dataset and the last client gets the remaining samples
    client_dataset_sizes = [len(dataset) // number_of_clients] * (number_of_clients - 1)
    client_dataset_sizes.append(len(dataset) - sum(client_dataset_sizes))

    # Splits the dataset into subsets for each client and returns the subsets
    return torch.utils.data.random_split(dataset, client_dataset_sizes)
