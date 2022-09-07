# Changelog

## 0.2.0

*Unreleased*

- Non-i.i.d dataset splitting was implemented
  - There are now four options for splitting datasets: **random**, **unbalanced labels**, **unbalanced sample count**, and **unbalanced**
  - **Random** is the same splitting strategy that was already present in the previous release and it is the default, splitting randomly results in an i.i.d. split of the dataset where all clients have the same amount of samples
  - **Unbalanced labels** distributes the samples of the dataset among the clients in a way such that each client gets the same amount of samples, but the labels are unbalanced, i.e., the number of samples per label differ, the label ratios of each client follow a Dirichlet distribution, the statistical heterogeneity level of the client data points can be controlled using a new command line argument
  - **Unbalanced sample count** distributes the samples of the dataset among the clients in a way such that clients have a different amount of samples, the amount of samples per client follows a log-normal distribution whose parameters can be controlled via a new command line argument
  - **Unbalanced** is a mix between the unbalanced labels and unbalanced sample counts splitting methods, the samples are distributed among the clients in such a way that the labels and the sample counts are unbalanced, where the label ratios follow a Dirichlet distribution and the sample counts of the clients follow a log-normal distribution, the parameters for both distributions can be controlled via the same new command line arguments that were introduced for the unbalanced label and the unbalanced sample count dataset splitting strategies
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
