"""Contains the actual entrypoint to the application."""

import sys
import logging

import torch

from fl.models import LeNet5
from fl.datasets import load_cifar_10, split_dataset
from fl.federated_learning import FederatedLearningCentralServer, FederatedLearningClient


class Application:
    """Represents the federated learning application."""


    def __init__(self) -> None:
        """Initializes a new Application instance."""

        self.logger = logging.getLogger('fl')
        self.logger.setLevel(logging.DEBUG)
        logging_formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
        console_logging_handler = logging.StreamHandler(sys.stdout)
        console_logging_handler.setLevel(logging.DEBUG)
        console_logging_handler.setFormatter(logging_formatter)
        self.logger.addHandler(console_logging_handler)

    def run(self) -> None:
        """Runs the application."""

        # Selects the device the training and validation will be performed on
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Loading the datasets
        self.logger.info('Loading datasets...')
        number_of_clients = 10
        training_subset, validation_subset = load_cifar_10('./datasets/cifar-10')
        client_subsets = split_dataset(training_subset, number_of_clients)

        # Creates the clients
        self.logger.info('Creating %d clients...', number_of_clients)
        clients = []
        for index in range(number_of_clients):
            client_local_model = LeNet5(input_shape=training_subset[0][0].shape)
            clients.append(FederatedLearningClient(device, client_local_model, client_subsets[index]))

        # Creates the central server
        global_model = LeNet5(input_shape=training_subset[0][0].shape)
        central_server = FederatedLearningCentralServer(clients, device, global_model, validation_subset)

        # Performs the federated training for the specified number of communication rounds
        for communication_round in range(1, 11):
            self.logger.info('Starting communication round %d...', communication_round)
            central_server.train_clients_and_update_global_model(5)
            validation_loss, validation_accuracy = central_server.validate()
            self.logger.info(
                'Finished communication round %d, validation loss: %f, validation accuracy: %f',
                communication_round,
                validation_loss,
                validation_accuracy
            )

        # Saves the trained global to disk
        self.logger.info('Finished federated training, saving trained global model to disk...')
        central_server.save_checkpoint('./trained-global-model.pt')
