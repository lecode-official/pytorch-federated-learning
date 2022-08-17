# Federated Learning in PyTorch

This is an implementation of federated learning using federated averaging (FedAvg) introduced by McMahan et al. in [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629).

The goal of this implementation is to simulate federated learning on an arbitrary number of clients using different models and datasets, which can form the basis of federated learning experiments. To simplify things, the models are all classifiers, which are trained on classic vision datasets.

## Getting Started

In order to get the federated learning simulator and run it, you need to install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and, if necessary, [Git](https://git-scm.com/downloads). After that, you are ready to clone the project:

```bash
git clone https://github.com/lecode-official/pytorch-federated-learning.git
cd pytorch-federated-learning/source
```

Before running the federated learning simulator, all dependencies have to be installed, which can easily be achieved using Miniconda:

```bash
conda env create -f environment.yaml
```

To use the virtual environment it must be activated first. After using the environment it has to be deactivated:

```bash
conda activate fad
python -m fl <command> <arguments...>
conda deactivate
```

When you install new packages, please update the environment file like so:

```bash
conda env export | grep -v "prefix" > environment.yaml
```

When someone else has added dependencies to the environment, then you can update your environment from the `environment.yaml` like so (the `--prune` switch makes sure that dependencies that have been removed from the `environment.yaml` are uninstalled):

```bash
conda env update --file environment.yaml --prune
```

## Training Models

To train models using federated learning or to perform baseline experiments, which can be used to compare federated learning results to, you can use the `fl` package. The `fl` package supports multiple commands. If you want to train a model using federated averaging, you can use the `federated-averaging` command. You have to specify the model that you want to train, the dataset that you want to train the model on, the path to the dataset, the number of communication rounds, the number of clients, and the path to the directory into which the global model checkpoints, hyperparameters, and the training statistics will be saved. The command will save the hyperparameters in a YAML file, and the training statistics of the central server and the clients in two separate CSV files. Furthermore, the command saves a global model checkpoint every time the validation accuracy of it outperforms all past global models. In order to not overcrowd the output directory, only the last 5 checkpoints are being retained and older ones are deleted. At the end of the training, the final model will be saved as well. The number of checkpoint files that are being retained can be configured using the `--number-of-checkpoint-files-to-retain`/`-R` argument.

The following example will train a LeNet-5 on MNIST using federated averaging for 20 communication rounds with 10 clients and saves the resulting global model checkpoint files, hyperparameters, and training statistics into the directory `./experiments/fedavg`. The dataset is expected to be in the `./datasets/mnist` directory. If it does not exist, yet, it is automatically downloaded.

```bash
python -m fl federated-averaging \
    --model lenet-5 \
    --dataset mnist \
    --dataset-path ./datasets/mnist \
    --number-of-clients 10 \
    --number-of-communication-rounds 20 \
    --output-path ./experiments/fedavg
```

Currently the following models and datasets are supported:

**Models:**

- LeNet-5 (`--model lenet-5`)

**Datasets:**

- MNIST (`--dataset mnist`)
- CIFAR-10 (`--dataset cifar-10`)

The training hyperparameters can be specified using the arguments `--learning-rate`/`-l`, `--momentum`/`-M`, `--weight-decay`/`-w`, and `--batch-size`/`-b`. Using the argument `--number-of-local-epochs`/`-e`, the number of epochs for which each client trains before sending the model updates back to the central server, can be specified. If you want the training to only run on the CPU (e.g., to better the debug the application), you can use the `--force-cpu`/`-c` flag. To use client sub-sampling, i.e., only using a subset of the total client population for each communication round, you can use the `--number-of-clients-per-communication-round`/`-N` argument. If not specified, the argument defaults to the number of clients, i.e., all clients are used for training in every communication round.

To train a baseline model without using federated learning, you can use the `baseline` command. You have to specify the model, the dataset, the dataset path, the number of epochs, and the output path. Again, you can specify the training hyperparameters using the arguments `--learning-rate`/`-l`, `--momentum`/`-M`, `--weight-decay`/`-w`, and `--batch-size`/`-b`. Also, you can specify the number of retained checkpoint files using the `--number-of-checkpoint-files-to-retain`/`-R` argument and force training on the CPU using the `--force-cpu`/`-c` flag.

The following example will train a LeNet-5 on MNIST for 25 epochs and saves the resulting model checkpoint files, hyperparameters, and training statistics into the `./experiments/baseline` directory. The dataset is expected to be in the `./datasets/mnist` directory. Again, if it does not exist, yet, it is automatically downloaded.

```bash
python -m fl baseline \
    --model lenet-5 \
    --dataset mnist \
    --dataset-path ./datasets/mnist \
    --number-of-epochs 25 \
    --output-path ./experiments/baseline
```

Finally, the training statistics of any experiment can be plotted using the `plot-training-statistics` command. It requires the path to the directory that contains the hyperparameters file and the training statistics files of the experiment as well as the path to the file into which the generated plot is to be saved. The following example plots the training results of the federated averaging and baseline experiments from above:

```bash
python -m fl plot-training-statistics \
    ./experiments/fedavg \
    ./fedavg-training-statistics.png

python -m fl plot-training-statistics \
    ./experiments/baseline \
    ./baseline-training-statistics.png
```

For baseline experiments, the training and validation accuracy and loss are being plotted. For federated learning experiments, the validation accuracy and loss of the central server as well as the training accuracy and loss of the clients are plotted. Since federated learning experiments may have a large client population, the number of clients that are plotted is restricted. By default, the first 100 clients are plotted. Using the `--maximum-number-of-clients-to-plot`/`-n` argument, the number of plotted clients can be configured. The `--client-sampling-method`/`-s` specifies how the clients are being sampled: when specifying `first` the first 100 clients are plotted and when specifying `random` 100 random clients are being plotted. Finally, the `--font`/`-f` argument can be used to specify whether a serif or a sans-serif font will be used for rendering plots. A serif font is ideal when embedding the plot in a LaTeX document and a sans-serif font is recommended for other types of documents such as presentations. By default a serif font is used.

## Preliminary Experiment Results

### Baseline Experiments

| Model   | Dataset | Learning Rate | Momentum | Weight Decay | Batch Size | Epochs | Best Validation Accuracy |
|---------|---------|--------------:|---------:|-------------:|-----------:|-------:|-------------------------:|
| LeNet-5 | MNIST   | 0.01          | 0.9      | 0.0005       | 128        | 50     | 99.09% (Epoch 44)        |

### Federated Averaging Experiments

| Model   | Dataset | Samples per Client | Learning Rate | Momentum | Weight Decay | Batch Size | Clients | Communication Rounds | Clients per Communication Round | Local Epochs | Best Validation Accuracy |
|---------|---------|-------------------:|--------------:|---------:|-------------:|-----------:|--------:|---------------------:|--------------------------------:|-------------:|-------------------------:|
| LeNet-5 | MNIST   | 60                 | 0.01          | 0.9      | 0.0005       | 60         | 1000    | 100                  | 100                             | 5            | 93.06% (Epoch 100)       |
| LeNet-5 | MNIST   | 600                | 0.01          | 0.9      | 0.0005       | 16         | 100     | 20                   | 50                              | 5            | 98.48% (Epoch 20)        |

## Contributing

If you'd like to contribute, there are multiple ways you can help out. If you find a bug or have a feature request, please feel free to open an issue on [GitHub](https://github.com/lecode-official/pytorch-federated-learning/issues). If you want to contribute code, please fork the repository and use a feature branch. Pull requests are always welcome. Before forking, please open an issue where you describe what you want to do. This helps to align your ideas with mine and may prevent you from doing work, that I am already planning on doing. If you have contributed to the project, please add yourself to the [contributors list](CONTRIBUTORS.md) and add all your changes to the [changelog](CHANGELOG.md). To help speed up the merging of your pull request, please comment and document your code extensively and try to emulate the coding style of the project.

## License

The code in this project is licensed under the MIT license. For more information see the [license file](LICENSE).
