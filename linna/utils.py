import torch
import os
from torch import nn
import numpy as np
import ast
from torch.utils.data.dataloader import DataLoader

from linna.verification.nnet import NNet


def load_tf_network(file: str) -> torch.nn.Sequential:
    """
    Loads a TensorFlow network (``.tf`` file) and returns a PyTorch Sequential neural network

    Parameters
    ----------
    file: str
        File containing TensorFlow network (``.tf`` file)

    Returns
    -------
    torch.nn.Sequential
        Neural network

    """
    activation = None
    layers = []
    weight = None
    with open(file, "r") as f:
        for line in f.readlines():
            if line.startswith("["):
                t = torch.Tensor(ast.literal_eval(line))
                if len(t.size()) > 1:
                    weight = t
                else:
                    in_features, out_features = weight.size(1), weight.size(0)
                    linear = torch.nn.Linear(in_features=in_features, out_features=out_features)
                    with torch.no_grad():
                        linear.weight = torch.nn.Parameter(weight)
                        linear.bias = torch.nn.Parameter(t)
                    layers.append(linear)
                    if activation is not None:
                        layers.append(activation)
                    activation = None
            else:
                if line.startswith("ReLU"):
                    activation = torch.nn.ReLU()
    return torch.nn.Sequential(*layers)


def get_accuracy(loader: torch.utils.data.DataLoader, model: torch.nn.Sequential, size=None):
    """

    Parameters
    ----------
    loader: torch.utils.data.DataLoader
        Data loader
    model: torch.nn.Sequential
        Neural network
    size: Optional[int]
        Number of inputs to consider

    Returns
    -------
    float
        Accuracy of network

    """
    correct = 0
    total = 0
    with torch.no_grad():
        for idx, data in enumerate(loader):
            images, labels = data
            if size and idx * len(images) > size:
                break
            outputs = model(images.view(-1, 784))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def load_model(path):
    name = os.path.basename(path)
    return torch.load(path), "model-{}".format(name)


def load_experiment(path):
    with open(path, "r") as txt_file:
        l = txt_file.read().split("\n")
        validation_acc = l[2].replace("[", "").replace("]", "").split(",")
        train_acc = l[4].replace("[", "").replace("]", "").split(",")
        validation_acc = [float(e) for e in validation_acc]
        train_acc = [float(e) for e in train_acc]
        import os
        return validation_acc, train_acc, os.path.basename(path)


def save_results(accuracies, reduction_rates, file_name):
    with open(file_name, "w") as file:
        file.write("rr,acc\n")
        for r, a in zip(reduction_rates, accuracies):
            file.write(f"{r},{a}\n")


def get_counterexamples(original_model, reduced_model, loader, true_label=False):
    """
    Returns the counter examples (w.r.t. classification)

    :param original_model:
    :param reduced_model:
    :param true_label:
    :return:
    """
    counterexamples = []
    example_labels = []
    with torch.no_grad():
        for idx, data in enumerate(loader):
            images, labels = data
            original_out = original_model.forward(images.view(-1, 784))
            reduced_out = reduced_model.forward(images.view(-1, 784))
            _, original_predicted = torch.max(original_out.data, 1)
            _, reduced_predicted = torch.max(reduced_out.data, 1)
            if torch.all(original_predicted == reduced_predicted):
                continue
            else:
                counterexamples.append(images[original_predicted != reduced_predicted])
                example_labels.append(labels[original_predicted != reduced_predicted])
    if counterexamples:
        if true_label:
            return torch.cat(counterexamples), torch.cat(example_labels)
        return torch.cat(counterexamples)
    if true_label:
        return [], []
    return counterexamples


def forward(torch_model, X, layer_idx, grad=False):
    if grad:
        return torch_model[:(layer_idx + 1) * 2](X)
    else:
        with torch.no_grad():
            return torch_model[:(layer_idx + 1) * 2](X)


def nnet_to_torch(network: NNet):
    layers = []
    last_layer = len(network.weights) - 1
    for idx, (weight, bias) in enumerate(zip(network.weights, network.biases)):
        out_neurons, in_neurons = weight.shape
        linear_layer = nn.Linear(in_features=in_neurons, out_features=out_neurons)
        with torch.no_grad():
            linear_layer.weight = nn.Parameter(torch.Tensor(weight))
            linear_layer.bias = nn.Parameter(torch.Tensor(bias))
        layers.append(linear_layer)
        if idx < last_layer:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def is_real_cex(network, cex: torch.Tensor, target_cls: int):
    out = network.forward(cex).cpu().detach().numpy()
    max_classes = np.argwhere(out == np.max(out)).reshape(-1).tolist()
    return target_cls not in max_classes or len(max_classes) > 1