"""Datasets used to train federated models."""

import torch
import torchvision


AVAILABLE_DATASETS = ['mnist', 'cifar-10']
DEFAULT_DATASET = 'mnist'


def load_cifar_10(
        path: str,
        minimum_sample_size: tuple[int, int]
    ) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, tuple[int, int, int], int]:
    """Loads the training and validation subsets of the CIFAR-10 dataset.

    Args:
        path (str): The path to the directory that contains the CIFAR-10 dataset. If the dataset does not exists at that location, then it is
            downloaded automatically.
        minimum_sample_size (tuple[int, int]): The minimum height and minimum width of the samples. If required, the size of the samples is adapted.

    Returns:
        tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]: Returns a tuple containing the training and the validation subsets of the CIFAR-10
            dataset, as well as the shape of its samples, and the number of classes.
    """

    transforms = []
    sample_size = (32, 32)
    if sample_size[0] < minimum_sample_size[0] or sample_size[1] < minimum_sample_size[1]:
        sample_size = (max(sample_size[0], minimum_sample_size[0]), max(sample_size[1], minimum_sample_size[1]))
        transforms.append(torchvision.transforms.Resize(sample_size))
    transforms.append(torchvision.transforms.ToTensor())
    transforms.append(torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]))

    training_subset = torchvision.datasets.CIFAR10(root=path, train=True, download=True, transform=torchvision.transforms.Compose(transforms))
    validation_subset = torchvision.datasets.CIFAR10(root=path, train=False, download=True, transform=torchvision.transforms.Compose(transforms))

    return training_subset, validation_subset, (3, sample_size[0], sample_size[1]), 10


def load_mnist(
        path: str,
        minimum_sample_size: tuple[int, int]
    ) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, tuple[int, int, int], int]:
    """Loads the training and validation subsets of the MNIST dataset.

    Args:
        path (str): The path to the directory that contains the MNIST dataset. If the dataset does not exists at that location, then it is downloaded
            automatically.
        minimum_sample_size (tuple[int, int]): The minimum height and minimum width of the samples. If required, the size of the samples is adapted.

    Returns:
        tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]: Returns a tuple containing the training and the validation subsets of the MNIST
            dataset, as well as the shape of its samples, and the number of classes.
    """

    transforms = []
    sample_size = (28, 28)
    if sample_size[0] < minimum_sample_size[0] or sample_size[1] < minimum_sample_size[1]:
        sample_size = (max(sample_size[0], minimum_sample_size[0]), max(sample_size[1], minimum_sample_size[1]))
        transforms.append(torchvision.transforms.Resize(sample_size))
    transforms.append(torchvision.transforms.ToTensor())
    transforms.append(torchvision.transforms.Normalize(mean=[0.1307], std=[0.3081]))

    training_subset = torchvision.datasets.MNIST(root=path, train=True, download=True, transform=torchvision.transforms.Compose(transforms))
    validation_subset = torchvision.datasets.MNIST(root=path, train=False, download=True, transform=torchvision.transforms.Compose(transforms))

    return training_subset, validation_subset, (1, sample_size[0], sample_size[1]), 10


def create_dataset(
        dataset_type: str,
        path: str,
        minimum_sample_size: tuple[int, int]
    ) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, tuple[int, int, int], int]:
    """Creates the specified dataset.

    Args:
        dataset_type (str): The type of dataset that is to be created.
        path (str): The path to the directory that contains the dataset. If the dataset does not exist, it is automatically downloaded.
        minimum_sample_size (tuple[int, int]): The minimum height and minimum width of the samples. If required, the size of the samples is adapted.

    Raises:
        ValueError:
            If the specified dataset type is not supported or the path is None, an exception is raised.

    Returns:
        tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, tuple, int]: Returns a tuple containing the training subset and the validation
            subset of the created dataset, as well as the shape of its samples, and the number of classes.
    """

    if path is None:
        raise ValueError('No dataset path was specified.')

    if dataset_type == 'cifar-10':
        return load_cifar_10(path, minimum_sample_size)

    if dataset_type == 'mnist':
        return load_mnist(path, minimum_sample_size)

    raise ValueError(f'The dataset type "{dataset_type}" is not supported.')


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
