"""Contains the descriptor for the federated-averaging command."""

from argparse import ArgumentParser

from fl.models import ModelType
from fl.datasets import DatasetType
from fl.commands.base import BaseCommandDescriptor


class FederatedAveragingCommandDescriptor(BaseCommandDescriptor):
    """Represents the descriptor for the federated-averaging command."""

    def get_name(self) -> str:
        """Gets the name of the command.

        Returns:
            str: Returns the name of the command.
        """

        return 'federated-averaging'

    def get_description(self) -> str:
        """Gets the description of the command.

        Returns:
            str: Returns the description of the command.
        """

        return 'Performs federating learning using federated averaging (FedAvg).'

    def add_arguments(self, argument_parser: ArgumentParser) -> None:
        """Adds the command line arguments to the command line argument parser.

        Args:
            argument_parser (ArgumentParser): The command line argument parser to which the arguments are to be added.
        """

        argument_parser.add_argument(
            '-n',
            '--number-of-clients',
            dest='number_of_clients',
            type=int,
            default=10,
            help='The number of federated learning clients. Defaults to 10.'
        )
        argument_parser.add_argument(
            '-N',
            '--number-of-clients-per-communication-round',
            dest='number_of_clients_per_communication_round',
            type=int,
            default=None,
            help='''One of the primary bottlenecks in the communication between the central server and its clients is the number of clients that the
                central server has to communicate with in each communication round. One easy method of reducing this overhead, is to subsample the
                client population. In each communication round, the central server only selects a subset of clients, which will train and communicate
                their updates back. This parameter specifies the number of clients that will be selected at random in each communication round.
                Defaults to the number of clients.
            '''
        )
        argument_parser.add_argument(
            '-m',
            '--model',
            dest='model_type',
            type=str,
            choices=ModelType.available_models(),
            default=ModelType.default_model(),
            help=f'The model that is to be used for the training. Defaults to "{ModelType.default_model()}".'
        )
        argument_parser.add_argument(
            '-s',
            '--normalization-layer-kind',
            dest='normalization_layer_kind',
            type=str,
            choices=['group-normalization', 'batch-normalization'],
            default='group-normalization',
            help='''The kind of normalization layer that is to be used in the model (only supported for the some of the available model types). It is
                a known fact, that batch normalization highly depends on dataset statistics and therefore, generally, does not work in federated
                learning, because averaging the batch normalization layer parameters results in worse performance. Group normalization does not suffer
                from this problem, but generally has a lower performance than batch normalization in non-federated learning scenarios. When the client
                datasets are i.i.d. it is generally still possible to use batch normalization, but when the client datasets are highly non-i.i.d. it
                is advised to use group normalization. If the model was trained using group normalization, then this parameter should be set to
                "group-normalization", otherwise to "batch-normalization". Defaults to "group-normalization".
            '''
        )
        argument_parser.add_argument(
            '-d',
            '--dataset',
            dest='dataset_type',
            type=str,
            choices=DatasetType.available_datasets(),
            default=DatasetType.default_dataset(),
            help=f'The dataset that is to be used for the training. Defaults to "{DatasetType.default_dataset()}".'
        )
        argument_parser.add_argument(
            '-p',
            '--dataset-splitting-strategy',
            dest='dataset_splitting_strategy',
            type=str,
            choices=['random', 'unbalanced-labels', 'unbalanced-sample-count', 'unbalanced'],
            default='random',
            help='''Determines the way that the dataset is split up for the clients. The "random" strategy results in an i.i.d. split of the dataset
                where all clients have the same amount of samples. The "unbalanced-labels" strategy distributes the samples of the dataset among the
                clients in a way such that each client gets the same amount of samples, but the labels are unbalanced, i.e., the number of samples per
                label differs. The label ratios of each client follow a Dirichlet distribution, the statistical heterogeneity level of the client data
                points can be controlled using the --dirichlet-alpha argument. The "unbalanced-sample-count" strategy distributes the samples of the
                dataset among the clients in a way such that clients have a different amount of samples. The amount of samples per client follows a
                log-normal distribution whose parameters are controlled via the --log-normal-sigma argument. The "unbalanced" strategy is a mix
                between the "unbalanced-labels" and "unbalanced-sample-counts" splitting methods. The samples are distributed among the clients in
                such a way that the labels and the sample counts are unbalanced, where the label ratios follow a Dirichlet distribution and the sample
                counts of the clients follow a log-normal distribution. The parameters for both distributions can be controlled via the
                --dirichlet-alpha and the --log-normal-sigma arguments. Defaults to "random".
            '''
        )
        argument_parser.add_argument(
            '-S',
            '--log-normal-sigma',
            dest='log_normal_sigma',
            type=float,
            default=0.3,
            help='''The standard deviation of the normal distribution that underlies the log-normal distribution that is used to distribute the
                samples among the clients. Sigma controls how far to the right the distribution is skewed. As sigma approaches 0 it more and more
                resembles a normal distribution. As sigma gets larger the tail of the distribution to the right becomes larger. Defaults to 0.3.
            '''
        )
        argument_parser.add_argument(
            '-a',
            '--dirichlet-alpha',
            dest='dirichlet_alpha',
            type=float,
            default=0.3,
            help='''The concentration parameter for the Dirichlet distribution, which is used to distribute the labels among the clients. An alpha of
                1 results in a uniform distribution, i.e., all clients have approximately the same amount of samples per label. A value for alpha that
                is less than 1 results in a label distribution where each client concentrates on one label, i.e., each client has a lot of samples of
                one label and fewer samples of the other labels. A value for alpha that is greater than 1 results in a distribution were the clients
                have increasingly the same amount of samples per label. In simple terms, an alpha smaller than one results in a more heterogeneous
                split of the labels while a value greater than 1 results in a more homogeneous split of the labels. The smaller alpha gets, the more
                heterogeneous the split becomes, the greater alpha becomes, the more homogeneous the split becomes. Defaults to 0.3.
            '''
        )
        argument_parser.add_argument(
            '-D',
            '--dataset-path',
            dest='dataset_path',
            type=str,
            required=True,
            help='''The path to the directory that contains the dataset that is to be used for the training. If the dataset does not exist, it is
                downloaded automatically.
            '''
        )
        argument_parser.add_argument(
            '-r',
            '--number-of-communication-rounds',
            dest='number_of_communication_rounds',
            type=int,
            default=50,
            help='''The number of communication rounds of the federated learning. One communication round consists of sending the global model to the
                clients, instructing them to perform training on their local dataset, and aggregating their updated local models into a new global
                model. Defaults to 50.
            '''
        )
        argument_parser.add_argument(
            '-e',
            '--number-of-local-epochs',
            dest='number_of_local_epochs',
            type=int,
            default=5,
            help='The number of communication epochs for which the clients are training the model on their local data. Defaults to 5.'
        )
        argument_parser.add_argument(
            '-o',
            '--output-path',
            dest='output_path',
            type=str,
            required=True,
            help='''The path to the directory into which checkpoints of the trained global model as well as training statistics are to be stored.
                If the directory does not exist, it is created.
            '''
        )
        argument_parser.add_argument(
            '-R',
            '--number-of-checkpoint-files-to-retain',
            dest='number_of_checkpoint_files_to_retain',
            type=int,
            default=5,
            help='''After each communication round, the current state of the global model is saved to a checkpoint file if its validation accuracy is
                greater than any previous state of the global model. This can potentially result in a great number of checkpoint files that are being
                stored. This argument controls how many past checkpoint files are being retained. If the number of past checkpoint files exceeds this
                argument, then the oldest one is deleted. Defaults to 5.
            '''
        )
        argument_parser.add_argument(
            '-l',
            '--learning-rate',
            dest='learning_rate',
            type=float,
            default=0.01,
            help='''The initial learning rate of the optimizer. Depending on the --learning-rate-decay argument, the learning rate will be decayed
                during the training. Defaults to 0.01.
            '''
        )
        argument_parser.add_argument(
            '-L',
            '--learning-rate-decay',
            dest='learning_rate_decay',
            type=float,
            default=0.95,
            help='''The learning rate is decayed exponentially during the training. This argument is the decay rate of the learning rate. A decay rate
                1.0 would result in no decay at all. Defaults to 0.95.
            '''
        )
        argument_parser.add_argument(
            '-M',
            '--momentum',
            dest='momentum',
            type=float,
            default=0.9,
            help='The momentum of the optimizer. Defaults to 0.9.'
        )
        argument_parser.add_argument(
            '-w',
            '--weight-decay',
            dest='weight_decay',
            type=float,
            default=0.0005,
            help='The rate at which the weights are decayed during optimization. Defaults to 0.96.'
        )
        argument_parser.add_argument(
            '-b',
            '--batch-size',
            dest='batch_size',
            type=int,
            default=16,
            help='The size of mini-batches that are to be used during training and validation. Defaults to 16.'
        )
        argument_parser.add_argument(
            '-c',
            '--force-cpu',
            dest='force_cpu',
            action='store_true',
            help='Always use the CPU for training, even when a GPU is available.'
        )
