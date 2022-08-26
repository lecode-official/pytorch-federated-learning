#!/bin/bash

DATASETS_PATH=../datasets
OUTPUT_PATH=../experiments

# Parses the command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--datasets-path)
            DATASETS_PATH=$2
            shift
            shift
            ;;
        -o|--output-path)
            OUTPUT_PATH=$2
            shift
            shift
            ;;
    esac
done

# Makes sure that datasets path and the output path do not end with a slash
if [[ $DATASETS_PATH == */ ]]; then
    DATASETS_PATH=${DATASETS_PATH::-1}
fi
if [[ $OUTPUT_PATH == */ ]]; then
    OUTPUT_PATH=${OUTPUT_PATH::-1}
fi

# Baseline experiment for LeNet-5 trained on MNIST
python -m fl baseline \
    --model lenet-5 \
    --dataset mnist \
    --dataset-path $DATASETS_PATH/mnist \
    --number-of-epochs 50 \
    --learning-rate-decay 1.0 \
    --output-path $OUTPUT_PATH/$(date +"%Y-%m-%d-%H-%M-%S")-baseline-lenet-5-mnist

# Baseline experiment for LeNet-5 trained on CIFAR-10
python -m fl baseline \
    --model lenet-5 \
    --dataset cifar-10 \
    --dataset-path $DATASETS_PATH/cifar-10 \
    --number-of-epochs 100 \
    --learning-rate 0.1 \
    --learning-rate-decay 0.98 \
    --weight-decay 0.0025 \
    --output-path $OUTPUT_PATH/$(date +"%Y-%m-%d-%H-%M-%S")-baseline-lenet-5-cifar-10

# Baseline experiment for VGG11 trained on MNIST
python -m fl baseline \
    --model vgg11 \
    --normalization-layer-kind batch-normalization \
    --dataset mnist \
    --dataset-path $DATASETS_PATH/mnist \
    --number-of-epochs 50 \
    --learning-rate-decay 0.98 \
    --output-path $OUTPUT_PATH/$(date +"%Y-%m-%d-%H-%M-%S")-baseline-vgg11-batch-normalization-mnist
python -m fl baseline \
    --model vgg11 \
    --normalization-layer-kind group-normalization \
    --dataset mnist \
    --dataset-path $DATASETS_PATH/mnist \
    --number-of-epochs 50 \
    --learning-rate-decay 0.98 \
    --output-path $OUTPUT_PATH/$(date +"%Y-%m-%d-%H-%M-%S")-baseline-vgg11-group-normalization-mnist

# Baseline experiment for VGG11 trained on CIFAR-10
python -m fl baseline \
    --model vgg11 \
    --normalization-layer-kind batch-normalization \
    --dataset cifar-10 \
    --dataset-path $DATASETS_PATH/cifar-10 \
    --number-of-epochs 50 \
    --learning-rate-decay 0.98 \
    --output-path $OUTPUT_PATH/$(date +"%Y-%m-%d-%H-%M-%S")-baseline-vgg11-batch-normalization-cifar-10
python -m fl baseline \
    --model vgg11 \
    --normalization-layer-kind group-normalization \
    --dataset cifar-10 \
    --dataset-path $DATASETS_PATH/cifar-10 \
    --number-of-epochs 50 \
    --learning-rate-decay 0.98 \
    --output-path $OUTPUT_PATH/$(date +"%Y-%m-%d-%H-%M-%S")-baseline-vgg11-group-normalization-cifar-10

# Federated averaging experiments for LeNet-5 trained on MNIST
python -m fl federated-averaging \
    --model lenet-5 \
    --dataset mnist \
    --dataset-path $DATASETS_PATH/mnist \
    --number-of-clients 100 \
    --number-of-communication-rounds 100 \
    --number-of-clients-per-communication-round 10 \
    --learning-rate-decay 1.0 \
    --batch-size 60 \
    --output-path $OUTPUT_PATH/$(date +"%Y-%m-%d-%H-%M-%S")-fedavg-lenet-5-mnist-100-clients
python -m fl federated-averaging \
    --model lenet-5 \
    --dataset mnist \
    --dataset-path $DATASETS_PATH/mnist \
    --number-of-clients 1000 \
    --number-of-communication-rounds 200 \
    --number-of-clients-per-communication-round 100 \
    --learning-rate-decay 1.0 \
    --batch-size 60 \
    --output-path $OUTPUT_PATH/$(date +"%Y-%m-%d-%H-%M-%S")-fedavg-lenet-5-mnist-1000-clients
python -m fl federated-averaging \
    --model lenet-5 \
    --dataset mnist \
    --dataset-path $DATASETS_PATH/mnist \
    --number-of-clients 10000 \
    --number-of-communication-rounds 300 \
    --number-of-clients-per-communication-round 100 \
    --learning-rate-decay 1.0 \
    --batch-size 6 \
    --output-path $OUTPUT_PATH/$(date +"%Y-%m-%d-%H-%M-%S")-fedavg-lenet-5-mnist-10000-clients

# Federated averaging experiments for LeNet-5 trained on CIFAR-10
python -m fl federated-averaging \
    --model lenet-5 \
    --dataset cifar-10 \
    --dataset-path $DATASETS_PATH/cifar-10 \
    --number-of-clients 100 \
    --number-of-communication-rounds 100 \
    --number-of-clients-per-communication-round 10 \
    --learning-rate 0.1 \
    --learning-rate-decay 0.98 \
    --weight-decay 0.0025 \
    --batch-size 50 \
    --output-path $OUTPUT_PATH/$(date +"%Y-%m-%d-%H-%M-%S")-fedavg-lenet-5-cifar-10-100-clients
python -m fl federated-averaging \
    --model lenet-5 \
    --dataset cifar-10 \
    --dataset-path $DATASETS_PATH/cifar-10 \
    --number-of-clients 1000 \
    --number-of-communication-rounds 200 \
    --number-of-clients-per-communication-round 100 \
    --learning-rate 0.1 \
    --learning-rate-decay 0.98 \
    --weight-decay 0.0025 \
    --batch-size 50 \
    --output-path $OUTPUT_PATH/$(date +"%Y-%m-%d-%H-%M-%S")-fedavg-lenet-5-cifar-10-1000-clients
python -m fl federated-averaging \
    --model lenet-5 \
    --dataset cifar-10 \
    --dataset-path $DATASETS_PATH/cifar-10 \
    --number-of-clients 10000 \
    --number-of-communication-rounds 300 \
    --number-of-clients-per-communication-round 100 \
    --learning-rate 0.1 \
    --learning-rate-decay 0.98 \
    --weight-decay 0.0025 \
    --batch-size 5 \
    --output-path $OUTPUT_PATH/$(date +"%Y-%m-%d-%H-%M-%S")-fedavg-lenet-5-cifar-10-10000-clients

# Federated averaging experiments for VGG11 trained on MNIST
python -m fl federated-averaging \
    --model vgg11 \
    --normalization-layer-kind batch-normalization \
    --dataset mnist \
    --dataset-path $DATASETS_PATH/mnist \
    --number-of-clients 100 \
    --number-of-communication-rounds 100 \
    --number-of-clients-per-communication-round 10 \
    --learning-rate-decay 0.98 \
    --batch-size 60 \
    --output-path $OUTPUT_PATH/$(date +"%Y-%m-%d-%H-%M-%S")-fedavg-vgg11-batch-normalization-mnist-100-clients
python -m fl federated-averaging \
    --model vgg11 \
    --normalization-layer-kind group-normalization \
    --dataset mnist \
    --dataset-path $DATASETS_PATH/mnist \
    --number-of-clients 100 \
    --number-of-communication-rounds 100 \
    --number-of-clients-per-communication-round 10 \
    --learning-rate-decay 0.98 \
    --batch-size 60 \
    --output-path $OUTPUT_PATH/$(date +"%Y-%m-%d-%H-%M-%S")-fedavg-vgg11-group-normalization-mnist-100-clients
python -m fl federated-averaging \
    --model vgg11 \
    --normalization-layer-kind batch-normalization \
    --dataset mnist \
    --dataset-path $DATASETS_PATH/mnist \
    --number-of-clients 1000 \
    --number-of-communication-rounds 200 \
    --number-of-clients-per-communication-round 100 \
    --learning-rate-decay 0.98 \
    --batch-size 60 \
    --output-path $OUTPUT_PATH/$(date +"%Y-%m-%d-%H-%M-%S")-fedavg-vgg11-batch-normalization-mnist-1000-clients
python -m fl federated-averaging \
    --model vgg11 \
    --normalization-layer-kind group-normalization \
    --dataset mnist \
    --dataset-path $DATASETS_PATH/mnist \
    --number-of-clients 1000 \
    --number-of-communication-rounds 200 \
    --number-of-clients-per-communication-round 100 \
    --learning-rate-decay 0.98 \
    --batch-size 60 \
    --output-path $OUTPUT_PATH/$(date +"%Y-%m-%d-%H-%M-%S")-fedavg-vgg11-group-normalization-mnist-1000-clients
python -m fl federated-averaging \
    --model vgg11 \
    --normalization-layer-kind batch-normalization \
    --dataset mnist \
    --dataset-path $DATASETS_PATH/mnist \
    --number-of-clients 10000 \
    --number-of-communication-rounds 300 \
    --number-of-clients-per-communication-round 100 \
    --learning-rate-decay 0.98 \
    --batch-size 6 \
    --output-path $OUTPUT_PATH/$(date +"%Y-%m-%d-%H-%M-%S")-fedavg-vgg11-batch-normalization-mnist-10000-clients
python -m fl federated-averaging \
    --model vgg11 \
    --normalization-layer-kind group-normalization \
    --dataset mnist \
    --dataset-path $DATASETS_PATH/mnist \
    --number-of-clients 10000 \
    --number-of-communication-rounds 300 \
    --number-of-clients-per-communication-round 100 \
    --learning-rate-decay 0.98 \
    --batch-size 6 \
    --output-path $OUTPUT_PATH/$(date +"%Y-%m-%d-%H-%M-%S")-fedavg-vgg11-group-normalization-mnist-10000-clients

# Federated averaging experiments for VGG11 trained on CIFAR-10
python -m fl federated-averaging \
    --model vgg11 \
    --normalization-layer-kind batch-normalization \
    --dataset cifar-10 \
    --dataset-path $DATASETS_PATH/cifar-10 \
    --number-of-clients 100 \
    --number-of-communication-rounds 100 \
    --number-of-clients-per-communication-round 10 \
    --learning-rate 0.1 \
    --learning-rate-decay 0.98 \
    --batch-size 50 \
    --output-path $OUTPUT_PATH/$(date +"%Y-%m-%d-%H-%M-%S")-fedavg-vgg11-batch-normalization-cifar-10-100-clients
python -m fl federated-averaging \
    --model vgg11 \
    --normalization-layer-kind group-normalization \
    --dataset cifar-10 \
    --dataset-path $DATASETS_PATH/cifar-10 \
    --number-of-clients 100 \
    --number-of-communication-rounds 100 \
    --number-of-clients-per-communication-round 10 \
    --learning-rate 0.1 \
    --learning-rate-decay 0.98 \
    --batch-size 50 \
    --output-path $OUTPUT_PATH/$(date +"%Y-%m-%d-%H-%M-%S")-fedavg-vgg11-group-normalization-cifar-10-100-clients
python -m fl federated-averaging \
    --model vgg11 \
    --normalization-layer-kind batch-normalization \
    --dataset cifar-10 \
    --dataset-path $DATASETS_PATH/cifar-10 \
    --number-of-clients 1000 \
    --number-of-communication-rounds 200 \
    --number-of-clients-per-communication-round 100 \
    --learning-rate 0.1 \
    --learning-rate-decay 0.98 \
    --batch-size 50 \
    --output-path $OUTPUT_PATH/$(date +"%Y-%m-%d-%H-%M-%S")-fedavg-vgg11-batch-normalization-cifar-10-1000-clients
python -m fl federated-averaging \
    --model vgg11 \
    --normalization-layer-kind group-normalization \
    --dataset cifar-10 \
    --dataset-path $DATASETS_PATH/cifar-10 \
    --number-of-clients 1000 \
    --number-of-communication-rounds 200 \
    --number-of-clients-per-communication-round 100 \
    --learning-rate 0.1 \
    --learning-rate-decay 0.98 \
    --batch-size 50 \
    --output-path $OUTPUT_PATH/$(date +"%Y-%m-%d-%H-%M-%S")-fedavg-vgg11-group-normalization-cifar-10-1000-clients
python -m fl federated-averaging \
    --model vgg11 \
    --normalization-layer-kind batch-normalization \
    --dataset cifar-10 \
    --dataset-path $DATASETS_PATH/cifar-10 \
    --number-of-clients 10000 \
    --number-of-communication-rounds 300 \
    --number-of-clients-per-communication-round 100 \
    --learning-rate 0.1 \
    --learning-rate-decay 0.98 \
    --batch-size 5 \
    --output-path $OUTPUT_PATH/$(date +"%Y-%m-%d-%H-%M-%S")-fedavg-vgg11-batch-normalization-cifar-10-10000-clients
python -m fl federated-averaging \
    --model vgg11 \
    --normalization-layer-kind group-normalization \
    --dataset cifar-10 \
    --dataset-path $DATASETS_PATH/cifar-10 \
    --number-of-clients 10000 \
    --number-of-communication-rounds 300 \
    --number-of-clients-per-communication-round 100 \
    --learning-rate 0.1 \
    --learning-rate-decay 0.98 \
    --batch-size 5 \
    --output-path $OUTPUT_PATH/$(date +"%Y-%m-%d-%H-%M-%S")-fedavg-vgg11-group-normalization-cifar-10-10000-clients
