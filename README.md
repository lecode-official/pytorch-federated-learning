# Federated Learning in PyTorch

This is an implementation of federated learning using federated averaging (FedAvg) introduced by McMahan et al. in [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629).

The goal of this implementation is to simulate federated learning on an arbitrary number of clients using different models and datasets, which can form the basis of federated learning experiments. To simplify things, the models are all classifiers, which are trained on classic vision datasets.

## Getting Started

In order to get the federated learning simulator and run it, you need to install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and, if necessary, [Git](https://git-scm.com/downloads). After that, you are ready to clone the project:

```bash
git clone https://github.com/lecode-official/pytorch-federated-learning.git
cd pytorch-federated-learning/source
```

Before running the federated learning simulator, all dependencies have to be installed, which can easily be achieved using Miniconda. There are different environment files for the different operating systems and platforms. Please select an environment file that fits your operating system and platform. Currently online Linux AMD64 and MacOS ARM64 are officially supported.

```bash
conda env create -f environment.<environment>.yaml
```

To use the virtual environment it must be activated first. After using the environment it has to be deactivated:

```bash
conda activate fl
python -m fl <command> <arguments...>
conda deactivate
```

When you install new packages, please update the environment file. Please make sure to either create a new environment file for your operating system and platform (i.e., choose a moniker in the format "<operating-system>-<platform>", e.g., "windows-amd64"), or overwrite the one that matches your operating system and platform. Ideally, try to update all supported environments if you plan on creating a pull request. The environment file can be updated like so:

```bash
conda env export | grep -v "prefix" > environment.<environment>.yaml
```

When someone else has added or removed dependencies from the environment, you have update your environment from the `environment.<environment>.yaml` as well. Again, please make sure to select the environment that fits your operating system and platform. The `--prune` switch makes sure that dependencies that have been removed from the `environment.<environment>.yaml` are uninstalled):

```bash
conda env update --file environment.<environment>.yaml --prune
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

## Experiment Results

In this section experiment results on all supported models, datasets, and algorithms is reported. In order to have a comparative baseline, all models were trained on all datasets using regular non-federated learning. These experiments are not designed to reach state-of-the-art results and use no *"fancy"* techniques like data augmentation, complex learning rate schedules, or elaborate hyperparameter tuning. Instead they are designed to be as simple as possible, so that the merits of the algorithms and the impact of the number of clients can be properly gauged without having to discount the effects or auxillary techniques. But in turn, this also means that some of the reported accuracies are well below what can be reached and has been reported in the literature. Nevertheless, each experiment has been performed multiple times and the best performance out of these tries has been documented here. The experiments can be re-created using the provided [script](source/experiments.sh).

Some of the models used in the experiments use BatchNorm. It is a well-known fact that BatchNorm layers do not work well in a federated learning context, as they learn the statistics of the local data-generating distribution, which can differ wildly between clients in non-i.i.d. settings. Although some papers prescribe different schemes, e.g., keeping the BatchNorm statistics local instead of averaging them, the experiments documented here, simply average the BatchNorm statistics. GroupNorm, an alternative normalization technique, has been shown to work better in non-i.i.d. settings. Therefore, all models that use normalization layers, a trained using both BatchNorm and GroupNorm, to show the differences between the two.

For the baseline experiments, the models were trained on the training subset and validated on the test subset of the respective dataset. For the federated averaging experiments, the training subset of the respective dataset was split randomly and equally among the clients. The global model of the central server was validated on the test subset of the respective dataset.

### Baseline Experiments

| Model   | Normalization  | Dataset  | Learning Rate | Learning Rate Decay | Momentum | Weight Decay | Batch Size | Epochs | Best Validation Accuracy |
|---------|----------------|----------|--------------:|--------------------:|---------:|-------------:|-----------:|-------:|-------------------------:|
| LeNet-5 | n/a            | MNIST    | 0.01          | *None*              | 0.9      | 0.0005       | 128        | 50     | 99.36% (Epoch 50)        |
| LeNet-5 | n/a            | CIFAR-10 | 0.1           | 0.98                | 0.9      | 0.0025       | 128        | 100    | 69.20% (Epoch 98)        |
| VGG11   | BatchNorm      | MNIST    | 0.01          | 0.98                | 0.9      | 0.0005       | 128        | 50     | 99.65% (Epoch 45)        |
| VGG11   | GroupNorm (32) | MNIST    | 0.01          | 0.98                | 0.9      | 0.0005       | 128        | 50     | 99.63% (Epoch 32)        |
| VGG11   | BatchNorm      | CIFAR-10 | 0.01          | 0.98                | 0.9      | 0.0005       | 128        | 50     | 85.79% (Epoch 44)        |
| VGG11   | GroupNorm (32) | CIFAR-10 | 0.01          | 0.98                | 0.9      | 0.0005       | 128        | 50     | 84.48% (Epoch 46)        |

### Federated Averaging Experiments

| Model   | Normalization | Dataset  | Samples per Client | Learning Rate | Learning Rate Decay | Momentum | Weight Decay | Batch Size | Clients | Communication Rounds | Clients per Communication Round | Local Epochs | Best Validation Accuracy         |
|---------|---------------|----------|-------------------:|--------------:|--------------------:|---------:|-------------:|-----------:|--------:|---------------------:|--------------------------------:|-------------:|---------------------------------:|
| LeNet-5 | n/a           | MNIST    | 600                | 0.01          | *None*              | 0.9      | 0.0005       | 60         | 100     | 100                  | 10                              | 5            | 98.83% (Communication Round 92)  |
| LeNet-5 | n/a           | MNIST    | 60                 | 0.01          | *None*              | 0.9      | 0.0005       | 60         | 1000    | 200                  | 100                             | 5            | 96.60% (Communication Round 200) |
| LeNet-5 | n/a           | MNIST    | 6                  | 0.01          | *None*              | 0.9      | 0.0005       | 6          | 10000   | 300                  | 100                             | 5            | 95.47% (Communication Round 300) |
| LeNet-5 | n/a           | CIFAR-10 | 500                | 0.1           | 0.98                | 0.9      | 0.0025       | 60         | 100     | 100                  | 10                              | 5            | 57.56% (Communication Round 93)  |
| LeNet-5 | n/a           | CIFAR-10 | 50                 | 0.1           | 0.98                | 0.9      | 0.0025       | 60         | 1000    | 200                  | 100                             | 5            | 48.17% (Communication Round 197) |
| LeNet-5 | n/a           | CIFAR-10 | 5                  | 0.1           | 0.98                | 0.9      | 0.0025       | 6          | 10000   | 300                  | 100                             | 5            | 46.26% (Communication Round 299) |

## Contributing

If you'd like to contribute, there are multiple ways you can help out. If you find a bug or have a feature request, please feel free to open an issue on [GitHub](https://github.com/lecode-official/pytorch-federated-learning/issues). If you want to contribute code, please fork the repository and use a feature branch. Pull requests are always welcome. Before forking, please open an issue where you describe what you want to do. This helps to align your ideas with mine and may prevent you from doing work, that I am already planning on doing. If you have contributed to the project, please add yourself to the [contributors list](CONTRIBUTORS.md) and add all your changes to the [changelog](CHANGELOG.md). To help speed up the merging of your pull request, please comment and document your code extensively and try to emulate the coding style of the project.

## License

The code in this project is licensed under the MIT license. For more information see the [license file](LICENSE).
