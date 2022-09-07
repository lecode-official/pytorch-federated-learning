"""Contains the baseline command."""

import os
import csv
import signal
import logging
from datetime import datetime
from argparse import Namespace

import yaml
import torch

from fl.commands.base import BaseCommand
from fl.lifecycle import Trainer, Validator
from fl.datasets import DatasetType, create_dataset
from fl.models import ModelType, NormalizationLayerKind, create_model, get_minimum_input_size


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

        # Parses the model and dataset types
        model_type = ModelType(command_line_arguments.model_type)
        dataset_type = DatasetType(command_line_arguments.dataset_type)

        # Prepares the training statistics CSV file by writing the header to file
        with open(os.path.join(command_line_arguments.output_path, 'training-statistics.csv'), 'w') as training_statistics_file:
            csv_writer = csv.writer(training_statistics_file)
            csv_writer.writerow(['timestamp', 'epoch', 'training_loss', 'training_accuracy', 'validation_loss', 'validation_accuracy'])

        # Saves the hyperparameters for later reference
        with open(os.path.join(command_line_arguments.output_path, 'hyperparameters.yaml'), 'w') as hyperparameters_file:
            yaml.dump({
                'method': 'baseline',
                'model': model_type.value,
                'normalization_layer_kind': command_line_arguments.normalization_layer_kind,
                'dataset': dataset_type.value,
                'dataset_path': command_line_arguments.dataset_path,
                'number_of_epochs': command_line_arguments.number_of_epochs,
                'output_path': command_line_arguments.output_path,
                'number_of_checkpoint_files_to_retain': command_line_arguments.number_of_checkpoint_files_to_retain,
                'learning_rate': command_line_arguments.learning_rate,
                'learning_rate_decay': command_line_arguments.learning_rate_decay,
                'momentum': command_line_arguments.momentum,
                'weight_decay': command_line_arguments.weight_decay,
                'batch_size': command_line_arguments.batch_size,
                'force_cpu': command_line_arguments.force_cpu
            }, hyperparameters_file)

        # Selects the device the training and validation will be performed on
        device = 'cpu'
        device_name = 'CPU'
        if not command_line_arguments.force_cpu:
            if torch.cuda.is_available():
                device = 'cuda'
                device_name = torch.cuda.get_device_name(device)
            elif torch.backends.mps.is_available():
                device = 'mps'
                device_name = 'Apple Silicon GPU (MPS)'
        self.logger.info(f'Selected {device_name} to perform training...')

        # Loading the datasets
        self.logger.info('Loading dataset (%s)...', dataset_type.get_human_readable_name())
        minimum_sample_shape = get_minimum_input_size(model_type)
        training_subset, validation_subset, sample_shape, number_of_classes = create_dataset(
            dataset_type,
            command_line_arguments.dataset_path,
            minimum_sample_shape
        )

        # Creates the model
        self.logger.info('Creating model (%s)...', model_type.get_human_readable_name())
        normalization_layer_kind = NormalizationLayerKind(command_line_arguments.normalization_layer_kind)
        model = create_model(model_type, sample_shape, number_of_classes, normalization_layer_kind)

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
        current_learning_rate = command_line_arguments.learning_rate
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
            self.trainer.change_learning_rate(current_learning_rate)
            training_loss, training_accuracy = self.trainer.train_for_one_epoch()
            self.logger.info(
                'Finished training, training loss: %f, training accuracy: %f%%, learning rate: %f',
                training_loss,
                training_accuracy * 100,
                current_learning_rate
            )
            current_learning_rate = current_learning_rate * command_line_arguments.learning_rate_decay

            # Validates the updated model and reports its loss and accuracy
            self.logger.info(f'Validating model...')
            validation_loss, validation_accuracy = validator.validate()
            self.logger.info('Finished epoch %d, validation loss: %f, validation accuracy: %f%%', epoch, validation_loss, validation_accuracy * 100)

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
                        model_type,
                        dataset_type,
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
            model_type,
            dataset_type,
            epoch,
            validation_accuracy * 100,
            command_line_arguments.output_path
        )

    def save_model_checkpoint(self, model_type: ModelType, dataset_type: DatasetType, epoch: str, accuracy: float, output_path: str) -> str:
        """Saves the current state of the model to disk.

        Args:
            model_type (ModelType): The type of model that is being trained.
            dataset_type (DatasetType): The type of dataset that the model being trained on.
            epoch (str): The current epoch.
            accuracy (float): The accuracy of the model.
            output_path (str): The path to the directory into which the model checkpoint file is to be saved.

        Returns:
            str: Returns the path to the checkpoint file.
        """

        current_date_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        model_checkpoint_file_path = os.path.join(
            output_path,
            f'{current_date_time}-{model_type.value}-{dataset_type.value}-baseline-epoch-{epoch}-accuracy-{accuracy:.2f}.pt'
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
