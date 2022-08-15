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
python -m fl <arguments...>
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

To train models using federated learning and federated averaging, you can use the `fl` package. You have to specify the model that you want to train, the dataset that you want to train the model on, the path to the dataset, the number of communication rounds, the number of clients, and the path to the file into which the trained global model will be saved after the training. The following command will train a LeNet-5 on MNIST for 20 communication rounds with 10 clients and save the resulting global model into a file called `lenet-5-mnist-trained-global-model.pt`. The dataset is expected to be in the `./datasets/mnist` directory. If it does not exist, yet, it is automatically downloaded.

```bash
python -m fl \
    --model lenet-5
    --dataset mnist \
    --dataset-path ./datasets/mnist \
    --number-of-clients 10 \
    --number-of-communication-rounds 20 \
    --model-output-file-path lenet-5-mnist-trained-global-model.pt \
```

Currently the following models and datasets are supported:

**Models:**

- LeNet-5 (`--model lenet-5`)

**Datasets:**

- MNIST (`--dataset mnist`)
- CIFAR-10 (`--model cifar-10`)

The training hyperparameters can be specified using the arguments `--learning-rate`/`-l`, `--momentum`/`-M`, `--weight-decay`/`-w`, and `--batch-size`/`-b`. Using the argument `--number-of-local-epochs`/`-e`, the number of epochs for which each client trains before sending the model updates back to the central server, can be specified. If you want the training to only run on the CPU (e.g., to better the debug the application), you can use the `--cpu`/`-c` flag. To use client sub-sampling, i.e., only using a subset of the total client population for each communication round, you can use the `--number-of-clients-per-communication-round`/`-N` argument. If not specified, the argument defaults to the number of clients, i.e., all clients are used for training in every communication round. Finally, it is possible to plot the training statistics (loss and accuracy) for the central server and all clients by passing a file path to the `--training-statistics-plot-output-file-path`/`p` argument. Be aware, though, that the plotting can take a long time for many clients, therefore, when using more than 250 clients, a warning is logged, and when using more than 1,000 clients, an error is raised and the application exists.

## Contributing

If you'd like to contribute, there are multiple ways you can help out. If you find a bug or have a feature request, please feel free to open an issue on [GitHub](https://github.com/lecode-official/pytorch-federated-learning/issues). If you want to contribute code, please fork the repository and use a feature branch. Pull requests are always welcome. Before forking, please open an issue where you describe what you want to do. This helps to align your ideas with mine and may prevent you from doing work, that I am already planning on doing. If you have contributed to the project, please add yourself to the [contributors list](CONTRIBUTORS.md) and add all your changes to the [changelog](CHANGELOG.md). To help speed up the merging of your pull request, please comment and document your code extensively and try to emulate the coding style of the project.

## License

The code in this project is licensed under MIT license. For more information see the [license file](LICENSE).
