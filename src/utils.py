import torch
import os
from torch import nn
import numpy as np
import ast
from torch.utils.data.dataloader import DataLoader



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


def writeNNet(weights, biases, inputMins, inputMaxes, means, ranges, fileName):
    '''
    Write network data to the .nnet file format
    Args:
        weights (list): Weight matrices in the network order
        biases (list): Bias vectors in the network order
        inputMins (list): Minimum values for each input
        inputMaxes (list): Maximum values for each input
        means (list): Mean values for each input and a mean value for all outputs. Used to normalize inputs/outputs
        ranges (list): Range values for each input and a range value for all outputs. Used to normalize inputs/outputs
        fileName (str): File where the network will be written
    '''

    # Open the file we wish to write
    with open(fileName, 'w') as f2:

        #####################
        # First, we write the header lines:
        # The first line written is just a line of text
        # The second line gives the four values:
        #     Number of fully connected layers in the network
        #     Number of inputs to the network
        #     Number of outputs from the network
        #     Maximum size of any hidden layer
        # The third line gives the sizes of each layer, including the input and output layers
        # The fourth line gives an outdated flag, so this can be ignored
        # The fifth line specifies the minimum values each input can take
        # The sixth line specifies the maximum values each input can take
        #     Inputs passed to the network are truncated to be between this range
        # The seventh line gives the mean value of each input and of all outputs
        # The eighth line gives the range of each input and of all outputs
        #     These two lines are used to map raw inputs to the 0 mean, unit range of the inputs and outputs
        #     used during training
        # The ninth line begins the network weights and biases
        ####################
        f2.write("// Neural Network File Format by Kyle Julian, Stanford 2016\n")

        # Extract the necessary information and write the header information
        numLayers = len(weights)
        inputSize = weights[0].shape[1]
        outputSize = len(biases[-1])
        maxLayerSize = inputSize

        # Find maximum size of any hidden layer
        for b in biases:
            if len(b) > maxLayerSize:
                maxLayerSize = len(b)

        # Write data to header
        f2.write("%d,%d,%d,%d,\n" % (numLayers, inputSize, outputSize, maxLayerSize))
        f2.write("%d," % inputSize)
        for b in biases:
            f2.write("%d," % len(b))
        f2.write("\n")
        f2.write("0,\n")  # Unused Flag

        # Write Min, Max, Mean, and Range of each of the inputs and outputs for normalization
        f2.write(','.join(str(inputMins[i]) for i in range(inputSize)) + ',\n')  # Minimum Input Values
        f2.write(','.join(str(inputMaxes[i]) for i in range(inputSize)) + ',\n')  # Maximum Input Values
        f2.write(','.join(str(means[i]) for i in range(inputSize + 1)) + ',\n')  # Means for normalizations
        f2.write(','.join(str(ranges[i]) for i in range(inputSize + 1)) + ',\n')  # Ranges for noramlizations

        ##################
        # Write weights and biases of neural network
        # First, the weights from the input layer to the first hidden layer are written
        # Then, the biases of the first hidden layer are written
        # The pattern is repeated by next writing the weights from the first hidden layer to the second hidden layer,
        # followed by the biases of the second hidden layer.
        ##################
        for w, b in zip(weights, biases):
            for i in range(w.shape[0]):
                for j in range(w.shape[1]):
                    f2.write(
                        "%.5e," % w[i][j])  # Five digits written. More can be used, but that requires more more space.
                f2.write("\n")

            for i in range(len(b)):
                f2.write("%.5e,\n" % b[i])  # Five digits written. More can be used, but that requires more more space.