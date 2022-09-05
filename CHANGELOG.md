# Changelog

## 0.2.0

*Unreleased*

- Non-i.i.d dataset splitting was implemented
  - There are now three options for splitting datasets: random, unbalanced labels, unbalanced client distribution
  - Random is the same splitting strategy that was already present in the previous release and it is the default
  - Unbalanced labels distributes the samples of the dataset among the clients in a way so that each client only gets a subset of all available labels, the number of labels that each client receives is determined by a Dirichlet distribution, the parameter of the Dirichlet distribution can be controlled via a new command line argument
  - Unbalanced client distribution distributes the samples of the dataset among the clients in a way such that clients have a different amount of samples, the amount of samples is normally distributed among the clients and the parameters of the normal distribution can be controlled via a new command line argument
- More experiments on all supported models and datasets using different dataset splitting strategies were performed and the results were recorded in the read me

## 0.1.0

Released on September 5, 2022

- Initial release
- Implements federated averaging using an arbitrary number of clients
- Implements a non-federated learning baseline to which federated learning algorithms can be compared
- Supports client sub-sampling, i.e., only a subset of all clients participates in each communication round
- Extensively logs hyperparameters and training statistics
- Intelligently retains model checkpoint files
- Training statistics can be plotted
- Supports the following models:
  - LeNet-5
  - VGG11
- Supports the following datasets:
  - MNIST
  - CIFAR-10
- Supports Linux on AMD64 and MacOS on ARM64
- Performed extensive experiments on all supported models and datasets using various numbers of clients and recorded the results in the read me
