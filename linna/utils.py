import torch
import os
from torch import nn
import numpy as np


def get_accuracy(loader, model, size=None):
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


def load_keras_model(path):
    keras_model = Keras_Model(filename=path)
    layers = []
    for idx, layer in enumerate(keras_model.model.layers):
        if len(layer.get_weights()) > 0:
            weight, bias = layer.get_weights()
            layers.append(nn.Linear(weight.shape[0], weight.shape[1]))
            with torch.no_grad():
                layers[-1].weight = nn.Parameter(torch.from_numpy(np.transpose(weight)))
                layers[-1].bias = nn.Parameter(torch.from_numpy(bias))
            if idx < len(keras_model.model.layers)-1:
                layers.append(nn.ReLU())
    torch_model = nn.Sequential(*layers)
    if sum(p.numel() for p in torch_model.parameters()) != keras_model.model.count_params():
        raise ValueError("Keras Model could not be loaded into PyTorch model")
    return torch_model


def export_keras_to_pb(source_path, target_path):
    keras_model = Keras_Model(filename=source_path)
    keras_model.model.save(target_path, save_format="tf",)


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
