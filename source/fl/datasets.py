"""Datasets used to train federated models."""

from enum import Enum
from typing import Type

import numpy
import torch
import torchvision


class DatasetType(Enum):
    """Represents the different types of datasets that are available."""

    MNIST = 'mnist'
    CIFAR_10 = 'cifar-10'

    @classmethod
    def available_datasets(cls: Type['DatasetType']) -> list[str]:
        """Retrieves a list of the values for the available datasets.

        Returns:
            list[str]: Returns a list that contains the values for all available datasets.
        """

        return [dataset_type.value for dataset_type in cls]

    @classmethod
    def default_dataset(cls: Type['DatasetType']) -> str:
        """Retrieves the value for the default dataset.

        Returns:
            str: Returns the value of the default dataset.
        """

        return DatasetType.MNIST.value

    def get_human_readable_name(self) -> str:
        """Retrieves a human-readable name for the dataset type.

        Returns:
            str: Returns a human-readable name for the dataset type.
        """

        human_readable_dataset_names_map = {
            DatasetType.MNIST: 'MNIST',
            DatasetType.CIFAR_10: 'CIFAR-10'
        }
        return human_readable_dataset_names_map[self]


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
        dataset_type: DatasetType,
        path: str,
        minimum_sample_size: tuple[int, int]
    ) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, tuple[int, int, int], int]:
    """Creates the specified dataset.

    Args:
        dataset_type (DatasetType): The type of dataset that is to be created.
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

    if dataset_type == DatasetType.MNIST:
        return load_mnist(path, minimum_sample_size)

    if dataset_type == DatasetType.CIFAR_10:
        return load_cifar_10(path, minimum_sample_size)

    raise ValueError(f'The dataset type "{dataset_type.value}" is not supported.')


def split_dataset_using_random_strategy(dataset: torch.utils.data.Dataset, number_of_clients: int) -> list[torch.utils.data.Dataset]:
    """Splits the specified dataset evenly among the specified number of clients. This results in an i.i.d. split of the dataset, where every client
    receives the same amount of samples.

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


def split_dataset_using_unbalanced_sample_counts_strategy(
        dataset: torch.utils.data.Dataset,
        number_of_clients: int,
        sigma: float
    ) -> list[torch.utils.data.Dataset]:
    """Splits the dataset among the clients in a way such that clients have a different amount of samples, the amount of samples per client follows a
    log-normal distribution. The implementation of this splitting strategy was inspired by the lognormal_unbalance_split dataset splitting function in
    FedLab (https://github.com/SMILELab-FL/FedLab/blob/master/fedlab/utils/dataset/functional.py#L54).

    Args:
        dataset (torch.utils.data.Dataset): The dataset that is to be split.
        number_of_clients (int): The number of clients among which the dataset is to be split.
        sigma (float): The standard deviation of the normal distribution that is underlying the log-normal distribution. Must be non-negative.

    Returns:
        tuple[torch.utils.data.Dataset, list[torch.utils.data.Dataset]]: Returns a list that contains subsets of the specified dataset for every
            federated learning client.
    """

    # The number of samples per client follows a log-normal distribution (the mean of the underlying normal distribution is set to the logarithm of
    # the average number of samples per client to make the algorithm numerically stable, because otherwise the values of the log-normal distribution
    # would explode and go to infinity if the average number of samples per client was large; taking the logarithm of the average number of samples
    # per client makes the values of the distribution much smaller, but the proportions between the clients stay the same and the values are then
    # normalized in the next step anyway, resulting in the exact same distribution of samples among the clients)
    number_of_samples = len(dataset)
    average_number_of_samples_per_client = number_of_samples / number_of_clients
    number_of_samples_per_client = numpy.random.lognormal(mean=numpy.log(average_number_of_samples_per_client), sigma=sigma, size=number_of_clients)
    number_of_samples_per_client = number_of_samples_per_client / number_of_samples_per_client.sum() * number_of_samples
    number_of_samples_per_client = number_of_samples_per_client.astype(int)

    # The stochastic nature of the process by which the number of samples per client was determined in the previous step means, that the sum of all
    # samples does not necessarily sum of the the total number of samples available (since the float values are truncated to integers, the sum should
    # be less than the number of samples available), therefore, a samples is added/subtracted from the first n clients, where n is the absolute
    # difference between the sum of the number of samples per client vs. the actual number of samples available
    difference = number_of_samples - number_of_samples_per_client.sum().item()
    if difference != 0:
        add_samples = difference > 0
        if add_samples:
            number_of_samples_per_client[:abs(difference)] += 1
        else:
            number_of_samples_per_client[:abs(difference)] -= 1

    # Splits the dataset into subsets for each client and returns the subsets
    return torch.utils.data.random_split(dataset, number_of_samples_per_client.tolist())
