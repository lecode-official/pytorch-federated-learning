"""Contains the plot-training-statistics command."""

import os
import random
import logging
from argparse import Namespace

import yaml
import pandas
import matplotlib
from tqdm import tqdm
from matplotlib import pyplot
from matplotlib.ticker import MaxNLocator

from fl.commands.base import BaseCommand


class PlotTrainingStatisticsCommand(BaseCommand):
    """Represents the plot-training-statistics command, which the training statistics of an experiment."""

    def __init__(self) -> None:
        """Initializes a new PlotTrainingStatisticsCommand instance."""

        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)

    def run(self, command_line_arguments: Namespace) -> None:
        """Runs the command.

        Args:
            command_line_arguments (Namespace): The parsed command line arguments.
        """

        # If the user specified to use a serif font, Matplotlib is configured to use a similar font to LaTeX, so that the figures are consistent with
        # LaTeX documents (a sans-serif font is the default of Matplotlib, so when the user specified to use a sans-serif font, nothing needs to be
        # done)
        if command_line_arguments.font == 'serif':
            matplotlib.rcParams['mathtext.fontset'] = 'stix'
            matplotlib.rcParams['font.family'] = 'STIXGeneral'

        # Loads the hyperparameters file of the experiment to determine the experiment method (for baseline experiments, only the training/validation
        # losses/accuracies of the training need to be plotted, while in the case of a federated learning experiment, the training statistics for the
        # central server and a subset of the clients need to be plotted)
        hyperparameters = None
        with open(os.path.join(command_line_arguments.experiment_path, 'hyperparameters.yaml'), 'r', encoding='utf-8') as hyperparameters_file:
            hyperparameters = yaml.load(hyperparameters_file, Loader=yaml.FullLoader)

        # Based on whether the experiment was a baseline or a federated learning experiment, different kinds of plots are generated
        if hyperparameters['method'] == 'baseline':
            self.plot_baseline_experiment_training_statistics(
                os.path.join(command_line_arguments.experiment_path, 'training-statistics.csv'),
                command_line_arguments.output_file_path
            )
        else:
            self.plot_federated_learning_experiment_training_statistics(
                os.path.join(command_line_arguments.experiment_path, 'central-server-training-statistics.csv'),
                os.path.join(command_line_arguments.experiment_path, 'client-training-statistics.csv'),
                command_line_arguments.output_file_path,
                hyperparameters['number_of_clients'],
                command_line_arguments.maximum_number_of_clients_to_plot,
                command_line_arguments.client_sampling_method
            )

    def plot_baseline_experiment_training_statistics(self, training_statistics_file_path: str, output_file_path: str) -> None:
        """Plots the training statistics of a baseline experiment.

        Args:
            training_statistics_file_path (str): The path to the file that contains the training statistics.
            output_file_path (str): The path to the file into which the generated plot is to be saved.
        """

        # Loads the training statistics of the experiment
        self.logger.info('Loading training statistics...')
        training_statistics = pandas.read_csv(training_statistics_file_path)
        maximum_loss = max(training_statistics['training_loss'].max(), training_statistics['validation_loss'].max())
        final_loss = training_statistics['validation_loss'].iloc[-1].item()
        final_accuracy = training_statistics['validation_accuracy'].iloc[-1].item() * 100

        # Creates the figure
        self.logger.info('Plotting training statistics...')
        figure = pyplot.figure(figsize=(15, 5), dpi=300)

        # Creates the plot for the validation loss/accuracy
        validation_accuracies_axis = figure.add_subplot(1, 2, 1)
        validation_accuracies_axis.set_title('Validation')
        validation_accuracies_axis.set_xlabel('Epoch')
        validation_accuracies_axis.set_ylabel('Accuracy')
        validation_accuracies_axis.set_ylim(0.0, 1.0)
        validation_accuracies_axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        validation_accuracies_axis.plot(
            training_statistics['epoch'].to_numpy(),
            training_statistics['validation_accuracy'].to_numpy(),
            color='blue',
            marker='.',
            markersize=5,
            label=f'Validation Accuracy (Final Accuracy: {final_accuracy:.2f}%)'
        )
        validation_losses_axis = validation_accuracies_axis.twinx()
        validation_losses_axis.set_ylabel('Loss')
        validation_losses_axis.set_ylim(0.0, maximum_loss)
        validation_losses_axis.plot(
            training_statistics['epoch'].to_numpy(),
            training_statistics['validation_loss'].to_numpy(),
            color='red',
            marker='.',
            markersize=5,
            label=f'Validation Loss (Final Loss: {final_loss:.2f})'
        )

        # Adds the legend to the validation loss and accuracy plot
        accuracy_handles, accuracy_labels = validation_accuracies_axis.get_legend_handles_labels()
        loss_handles, loss_labels = validation_losses_axis.get_legend_handles_labels()
        validation_accuracies_axis.legend(accuracy_handles + loss_handles, accuracy_labels + loss_labels)

        # Creates the plot for the training loss/accuracy
        training_accuracies_axis = figure.add_subplot(1, 2, 2)
        training_accuracies_axis.set_title('Training')
        training_accuracies_axis.set_xlabel('Epoch')
        training_accuracies_axis.set_ylabel('Accuracy')
        training_accuracies_axis.set_ylim(0.0, 1.0)
        training_accuracies_axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        training_accuracies_axis.plot(
            training_statistics['epoch'].to_numpy(),
            training_statistics['training_accuracy'].to_numpy(),
            color='blue',
            marker='.',
            markersize=5,
            label='Training Accuracy'
        )
        training_losses_axis = training_accuracies_axis.twinx()
        training_losses_axis.set_ylabel('Loss')
        training_losses_axis.set_ylim(0.0, maximum_loss)
        training_losses_axis.plot(
            training_statistics['epoch'].to_numpy(),
            training_statistics['training_loss'].to_numpy(),
            color='red',
            marker='.',
            markersize=5,
            label='Training Loss'
        )

        # Adds the legend to the training loss and accuracy plot
        accuracy_handles, accuracy_labels = training_accuracies_axis.get_legend_handles_labels()
        loss_handles, loss_labels = training_losses_axis.get_legend_handles_labels()
        training_accuracies_axis.legend(accuracy_handles + loss_handles, accuracy_labels + loss_labels)

        # Saves the generated plot
        self.logger.info('Saving generate training statistics plot...')
        figure.tight_layout()
        figure.savefig(output_file_path)

    def plot_federated_learning_experiment_training_statistics(
            self,
            central_server_training_statistics_file_path: str,
            client_training_statistics_file_path: str,
            output_file_path: str,
            number_of_clients: int,
            maximum_number_of_clients_to_plot: int,
            client_sampling_method: str) -> None:
        """Plots the training statistics of a federated learning experiment.

        Args:
            central_server_training_statistics_file_path (str): The path to the file that contains the training statistics of the central server.
            client_training_statistics_file_path (str): The path to the file that contains the training statistics of the clients.
            output_file_path (str): The path to the file into which the generated plot is to be saved.
            number_of_clients (int): The total number of clients that participated in the experiment.
            maximum_number_of_clients_to_plot (int): The maximum number of client training statistics that are to be plotted. Plotting too many
                clients may take too long and may result in a plot that is too large. This parameter makes sure that the number of clients plotted is
                capped at a reasonable number.
            client_sampling_method (str): Determines how the clients that are to be plotted are sampled. If there are more clients than the specified
                maximum number of clients, then only a subset is plotted (the size of that subset can be adjusted using the
                maximum_number_of_clients_to_plot parameter). This parameter determines whether a random subset or the first n elements are selected
                for plotting.
        """

        # Loads the training statistics of the experiment
        self.logger.info('Loading training statistics...')
        central_server_training_statistics = pandas.read_csv(central_server_training_statistics_file_path)
        client_training_statistics = pandas.read_csv(client_training_statistics_file_path)
        final_central_server_loss = central_server_training_statistics['validation_loss'].iloc[-1].item()
        final_central_server_accuracy = central_server_training_statistics['validation_accuracy'].iloc[-1].item() * 100

        # Plotting more than 100 clients becomes really slow and the resulting plots are gigantic, therefore, only the first 100 clients are plotted
        # if the number of clients exceeds 100
        client_ids_to_plot = None
        number_of_clients_to_plot = min(maximum_number_of_clients_to_plot, number_of_clients)
        if client_sampling_method == 'first':
            client_ids_to_plot = list(range(1, number_of_clients_to_plot + 1))
        else:
            client_ids_to_plot = random.sample(list(range(1, number_of_clients + 1)), number_of_clients_to_plot)

        # Creates the figure
        width, height = self.determine_optimal_grid_size(number_of_clients_to_plot, prefer_larger_width=True)
        figure = pyplot.figure(figsize=(int(2.5 * width), height), dpi=300)
        grid_specification = figure.add_gridspec(ncols=width + 1, nrows=height, width_ratios=[width] + [1] * width)

        # Determines the limits of the y-axis for the loss, so that the y-axes of the central server and the clients are all on the same scale (the
        # accuracy is bounded between 0 and 1, but the loss can grow almost arbitrarily)
        maximum_central_server_loss = central_server_training_statistics['validation_loss'].max()
        maximum_client_loss = client_training_statistics[[f'client_{client_id}_training_loss' for client_id in client_ids_to_plot]].max().max()
        maximum_loss = max(maximum_central_server_loss, maximum_client_loss)

        # Creates the plot for the validation loss and validation accuracy of the central server
        central_server_validation_accuracy_axis = figure.add_subplot(grid_specification[:, 0])
        central_server_validation_accuracy_axis.set_xlabel('Communication Rounds')
        central_server_validation_accuracy_axis.set_ylabel('Validation Accuracy')
        central_server_validation_accuracy_axis.set_title('Central Server')
        central_server_validation_accuracy_axis.set_ylim(0.0, 1.0)
        central_server_validation_accuracy_axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        central_server_validation_accuracy_axis.plot(
            central_server_training_statistics['communication_round'].to_numpy(),
            central_server_training_statistics['validation_accuracy'].to_numpy(),
            color='blue',
            linewidth=0.5,
            marker='.',
            markersize=8,
            label=f'Accuracy (Final Accuracy: {final_central_server_accuracy:.2f})'
        )
        central_server_validation_loss_axis = central_server_validation_accuracy_axis.twinx()
        central_server_validation_loss_axis.set_ylabel('Validation Loss')
        central_server_validation_loss_axis.set_ylim((0.0, maximum_loss))
        central_server_validation_loss_axis.plot(
            central_server_training_statistics['communication_round'].to_numpy(),
            central_server_training_statistics['validation_loss'].to_numpy(),
            color='red',
            linewidth=0.5,
            marker='.',
            markersize=8,
            label=f'Loss (Final Loss: {final_central_server_loss:.2f})'
        )

        # Adds the legend to the validation loss and accuracy plot
        accuracy_handles, accuracy_labels = central_server_validation_accuracy_axis.get_legend_handles_labels()
        loss_handles, loss_labels = central_server_validation_loss_axis.get_legend_handles_labels()
        central_server_validation_accuracy_axis.legend(accuracy_handles + loss_handles, accuracy_labels + loss_labels)

        # Creates the plots for the training loss and training accuracy of the clients
        with tqdm(total=number_of_clients_to_plot, desc='Plotting', unit='clients') as progress_bar:
            federated_learning_client_id_iterator = iter(client_ids_to_plot)
            for column in range(1, width + 1):
                for row in range(height):
                    federated_learning_client_id = next(federated_learning_client_id_iterator)
                    federated_learning_client_training_accuracy_axis = figure.add_subplot(grid_specification[row, column])
                    federated_learning_client_training_accuracy_axis.tick_params(
                        color='white',
                        left=False,
                        bottom=False,
                        labelleft=False,
                        labelbottom=False
                    )
                    federated_learning_client_training_accuracy_axis.set_ylim(0.0, 1.0)
                    federated_learning_client_training_accuracy_axis.text(
                        0.5,
                        0.1,
                        str(federated_learning_client_id),
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize='small',
                        transform=federated_learning_client_training_accuracy_axis.transAxes
                    )
                    federated_learning_client_training_accuracy_axis.plot(
                        client_training_statistics['communication_round'].to_numpy(),
                        client_training_statistics[f'client_{federated_learning_client_id}_training_accuracy'].to_numpy(),
                        color='blue',
                        linewidth=0.5,
                        marker='.',
                        markersize=4
                    )
                    federated_learning_client_training_loss_axis = federated_learning_client_training_accuracy_axis.twinx()
                    federated_learning_client_training_loss_axis.set_ylim((0.0, maximum_loss))
                    federated_learning_client_training_loss_axis.tick_params(right=False, labelright=False)
                    federated_learning_client_training_loss_axis.plot(
                        client_training_statistics['communication_round'].to_numpy(),
                        client_training_statistics[f'client_{federated_learning_client_id}_training_loss'].to_numpy(),
                        color='red',
                        linewidth=0.5,
                        marker='.',
                        markersize=4
                    )
                    progress_bar.update(1)

        # Creates an invisible axis, which is just a hack to place a title above the client plots
        federated_learning_client_title_axis = figure.add_subplot(grid_specification[:, 1:])
        federated_learning_client_title_axis.set_xticks([])
        federated_learning_client_title_axis.set_yticks([])
        federated_learning_client_title_axis.spines['right'].set_visible(False)
        federated_learning_client_title_axis.spines['top'].set_visible(False)
        federated_learning_client_title_axis.spines['bottom'].set_visible(False)
        federated_learning_client_title_axis.spines['left'].set_visible(False)
        federated_learning_client_title_axis.set_facecolor('none')
        federated_learning_client_title_axis.set_title('Clients')

        # Saves the generated plot
        self.logger.info('Saving generate training statistics plot...')
        figure.tight_layout()
        grid_specification.tight_layout(figure)
        figure.savefig(output_file_path)

    def determine_optimal_grid_size(self, number_of_elements: int, prefer_larger_width: bool) -> list[int]:
        """Determines the optimal edge lengths for a grid that should contain the specified number of elements. Each number of elements has multiple
        grids in which they can be arranged, but the optimal grid size is the one where both sides are as large as possible.

        Args:
            number (int): The number of elements that should be contained in the grid.
            prefer_larger_width (bool): For element counts that are not perfect squares, there are always two optimal grid sizes: one where the width
                is larger and one where the height is larger. This parameter controls which of the two is selected. If True, the grid size with a
                larger width is selected and if False, the grid size with a larger height is selected. For example, if there are 50 elements, then the
                optimal grid sizes are 5 by 10 and 10 by 5 elements. If larger widths are preferred, then 10 by 5 elements will be chosen as the
                optimal grid size, otherwise 5 by 10 will be chosen.

        Returns:
            tuple[int, int]: Returns the optimal grid size as tuple, where the first element is the width and the second element is the height.
        """

        # Grid sizes are always composed of widths and heights that are integer divisors of the number of elements, therefore all divisors of the
        # number of elements are determined
        grid_sizes = []
        for divisor_candidate in range(1, number_of_elements + 1):
            if number_of_elements % divisor_candidate == 0:
                grid_sizes.append((divisor_candidate, number_of_elements // divisor_candidate))

        # To find the optimal grid size, the grid sizes are ordered by their sum, the optimal grid sizes are always the ones where the sum of width
        # and height are lowest, if the number of elements is not a perfect square then there are always two optimal grid sizes, one where the width
        # is larger and one where the height is larger, therefore the grid sizes are then ordered by their width or height, which is controlled by the
        # prefer_larger_width parameter (if larger widths should be preferred, then the grid sizes are ordered by height, because the grid sizes are
        # ordered in ascending order and the grid size with the lower height will be first, the same applies accordingly when larger heights should be
        # preferred, in which case the grid sizes are ordered by width)
        grid_sizes = sorted(grid_sizes, key=lambda grid_size: (sum(grid_size), grid_size[1 if prefer_larger_width else 0]))

        # Now the optimal grid size is the first element in the list of all grid sizes
        return grid_sizes[0]
