"""Contains the descriptor for the plot-training-statistics command."""

from argparse import ArgumentParser

from fl.commands.base import BaseCommandDescriptor


class PlotTrainingStatisticsCommandDescriptor(BaseCommandDescriptor):
    """Represents the descriptor for the plot-training-statistics command."""

    def get_name(self) -> str:
        """Gets the name of the command.

        Returns:
            str: Returns the name of the command.
        """

        return 'plot-training-statistics'

    def get_description(self) -> str:
        """Gets the description of the command.

        Returns:
            str: Returns the description of the command.
        """

        return 'Plots the training statistics of an experiment.'

    def add_arguments(self, argument_parser: ArgumentParser) -> None:
        """Adds the command line arguments to the command line argument parser.

        Args:
            argument_parser (ArgumentParser): The command line argument parser to which the arguments are to be added.
        """

        argument_parser.add_argument(
            'experiment_path',
            type=str,
            help='The path to the directory that contains the experiment files.'
        )
        argument_parser.add_argument(
            'output_file_path',
            type=str,
            help='The path to the file into which the generated plot is to the saved.'
        )
        argument_parser.add_argument(
            '-f',
            '--font',
            dest='font',
            type=str,
            choices=['serif', 'sans-serif'],
            default='serif',
            help='''Determines whether a serif or a sans serif font will be used for rendering plots. A serif font is ideal when embedding the plot in
                a LaTeX document and a sans-serif font is recommended for other types of documents such as presentations. Defaults to "serif".
            '''
        )
        argument_parser.add_argument(
            '-n',
            '--maximum-number-of-clients-to-plot',
            dest='maximum_number_of_clients_to_plot',
            type=int,
            default=100,
            help='''The maximum number of client training statistics that are to be plotted. Plotting too many clients may take too long and may
                result in a plot that is too large. This argument makes sure that the number of clients plotted is capped at a reasonable number. Only
                applies to federated learning and not baseline experiments. Defaults to 100.
            '''
        )
        argument_parser.add_argument(
            '-s',
            '--client-sampling-method',
            dest='client_sampling_method',
            type=str,
            choices=['random', 'first'],
            default='first',
            help='''Determines how the clients that are to be plotted are sampled. If there are more clients than the specified maximum number of
                clients, then only a subset is plotted (the size of that subset can be adjusted using the --maximum-number-of-clients-to-plot
                argument). This argument determines whether a random subset or the first n elements are selected for plotting. Defaults to "first".
            '''
        )
