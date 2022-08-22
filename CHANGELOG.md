# Changelog

## 0.1.0

*Unreleased*

- Initial release
- Implements federated averaging using an arbitrary number of clients
- Implements a non-federated learning baseline to which federated learning algorithms can be compared
- Supports client sub-sampling, i.e., only a subset of all clients participates in each communication round
- Extensively logs hyperparameters and training statistics
- Intelligently retains model checkpoint files
- Training statistics can be plotted
- Supports the following models:
  - LeNet-5
- Supports the following datasets:
  - MNIST
  - CIFAR-10
- Supports Linux on AMD64 and MacOS on ARM64
