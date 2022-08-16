"""Contains the baseline command."""

import os
import csv
import signal
import logging
from datetime import datetime
from argparse import Namespace

import yaml
import torch

from fl.models import create_model
from fl.datasets import create_dataset
from fl.commands.base import BaseCommand
from fl.lifecycle import Trainer, Validator


class BaselineCommand(BaseCommand):
    """Represents the baseline command, which performs non-federated learning as a baseline to compare federated learning algorithms against."""

    def __init__(self) -> None:
        """Initializes a new BaselineCommand instance."""

        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)

        self.trainer = None
        self.is_aborting = False

    def run(self, command_line_arguments: Namespace) -> None:
        """Runs the command.

        Args:
            command_line_arguments (Namespace): The parsed command line arguments.
        """

        # Makes sure that the output directory exists
        os.makedirs(command_line_arguments.output_path, exist_ok=True)

        # Prepares the training statistics CSV file by writing the header to file
        with open(os.path.join(command_line_arguments.output_path, 'training-statistics.csv'), 'w') as training_statistics_file:
            csv_writer = csv.writer(training_statistics_file)
            csv_writer.writerow(['timestamp', 'epoch', 'training_loss', 'training_accuracy', 'validation_loss', 'validation_accuracy'])

        # Saves the hyperparameters for later reference
        with open(os.path.join(command_line_arguments.output_path, 'hyperparameters.yaml'), 'w') as hyperparameters_file:
            yaml.dump({
                'method': 'baseline',
                'model': command_line_arguments.model_type,
                'dataset': command_line_arguments.dataset_type,
                'dataset_path': command_line_arguments.dataset_path,
                'number_of_epochs': command_line_arguments.number_of_epochs,
                'output_path': command_line_arguments.output_path,
                'number_of_checkpoint_files_to_retain': command_line_arguments.number_of_checkpoint_files_to_retain,
                'learning_rate': command_line_arguments.learning_rate,
                'momentum': command_line_arguments.momentum,
                'weight_decay': command_line_arguments.weight_decay,
                'batch_size': command_line_arguments.batch_size,
                'force_cpu': command_line_arguments.force_cpu
            }, hyperparameters_file)

        # Selects the device the training and validation will be performed on
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if command_line_arguments.force_cpu:
            device = 'cpu'

        # Loading the datasets
        self.logger.info('Loading dataset (%s)...', command_line_arguments.dataset_type)
        training_subset, validation_subset, sample_shape, number_of_classes = create_dataset(
            command_line_arguments.dataset_type,
            command_line_arguments.dataset_path
        )

        # Creates the model
        self.logger.info('Creating model...')
        model = create_model(command_line_arguments.model_type, sample_shape, number_of_classes)

        # Creates the trainer
        self.logger.info('Creating trainer...')
        self.trainer = Trainer(
            device,
            model,
            training_subset,
            command_line_arguments.learning_rate,
            command_line_arguments.momentum,
            command_line_arguments.weight_decay,
            command_line_arguments.batch_size
        )

        # Creates the validator
        self.logger.info('Creating validator...')
        validator = Validator(
            device,
            model,
            validation_subset,
            command_line_arguments.batch_size
        )

        # Registers a signal handler, which graciously stops the training and saves the current state to disk, when the user hits Ctrl+C
        signal.signal(signal.SIGINT, lambda _, __: self.abort_training())

        # Performs the training for the specified number of epochs
        current_greatest_validation_accuracy = 0
        retained_model_checkpoint_file_paths = []
        for epoch in range(1, command_line_arguments.number_of_epochs + 1):

            # If the user hit Ctrl+C, then the training is aborted
            if self.is_aborting:
                self.logger.info('Graciously shutting down training... Hit Ctrl+C again to force quit...')
                break

            # Performs training on the model for one epoch
            self.logger.info('Starting epoch %d...', epoch)
            self.logger.info(f'Training model...')
            training_loss, training_accuracy = self.trainer.train_for_one_epoch()
            self.logger.info('Finished training, training loss: %f, training accuracy: %f', training_loss, training_accuracy * 100)

            # Validates the updated model and reports its loss and accuracy
            self.logger.info(f'Validating model...')
            validation_loss, validation_accuracy = validator.validate()
            self.logger.info('Finished epoch %d, validation loss: %f, validation accuracy: %f', epoch, validation_loss, validation_accuracy * 100)

            # Writes the training statistics into a CSV file
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open(os.path.join(command_line_arguments.output_path, 'training-statistics.csv'), 'a') as training_statistics_file:
                csv_writer = csv.writer(training_statistics_file)
                csv_writer.writerow([timestamp, epoch, training_loss, training_accuracy, validation_loss, validation_accuracy])

            # If the updated model has a better accuracy than any of its predecessors, then a checkpoint is saved for it, if the number of checkpoint
            # files that have already been saved, exceeds the number of checkpoint files to retain, then the oldest one is deleted (the checkpoint
            # file is not saved, if this is the last epoch, because the final model is saved anyway)
            if epoch != command_line_arguments.number_of_epochs:
                if validation_accuracy > current_greatest_validation_accuracy:

                    # Since the updated model outperformed all previous models, a checkpoint is saved for it
                    model_checkpoint_file_path = self.save_model_checkpoint(
                        command_line_arguments.model_type,
                        command_line_arguments.dataset_type,
                        epoch,
                        validation_accuracy * 100,
                        command_line_arguments.output_path
                    )
                    current_greatest_validation_accuracy = validation_accuracy

                    # If the number of saved checkpoint files exceeds the number of checkpoint files that should be retained, the oldest checkpoint
                    # file is deleted
                    retained_model_checkpoint_file_paths.append(model_checkpoint_file_path)
                    if len(retained_model_checkpoint_file_paths) > command_line_arguments.number_of_checkpoint_files_to_retain:
                        model_checkpoint_file_to_remove = retained_model_checkpoint_file_paths[0]
                        os.remove(model_checkpoint_file_to_remove)
                        retained_model_checkpoint_file_paths = retained_model_checkpoint_file_paths[1:]

        # Saves the trained model to disk
        self.logger.info('Finished training...')
        self.save_model_checkpoint(
            command_line_arguments.model_type,
            command_line_arguments.dataset_type,
            epoch,
            validation_accuracy * 100,
            command_line_arguments.output_path
        )

    def save_model_checkpoint(self, model_type: str, dataset_type: str, epoch: str, accuracy: float, output_path: str) -> str:
        """Saves the current state of the model to disk.

        Args:
            model_type (str): The type of model that is being trained.
            dataset_type (str): The type of dataset that the model being trained on.
            epoch (str): The current epoch.
            accuracy (float): The accuracy of the model.
            output_path (str): The path to the directory into which the model checkpoint file is to be saved.

        Returns:
            str: Returns the path to the checkpoint file.
        """

        model_checkpoint_file_path = os.path.join(
            output_path,
            f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}-{model_type}-{dataset_type}-baseline-{epoch}-epoch-{accuracy:.2f}-accuracy.pt'
        )
        self.logger.info('Saving model checkpoint to disk (%s)...', model_checkpoint_file_path)
        self.trainer.save_checkpoint(model_checkpoint_file_path)
        return model_checkpoint_file_path

    def abort_training(self) -> None:
        """Graciously aborts the federated learning."""

        # If the user hits Ctrl+C a second time, then the application is closed right away
        if self.is_aborting:
            exit()

        # Since this is the first time, that the user hit Ctrl+C, the aborting process is initiated
        self.is_aborting = True
        self.trainer.abort_training()
