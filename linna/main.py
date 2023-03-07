import argparse
from os.path import exists, isfile

import torch
from torchvision import datasets, transforms
from linna.abstraction import Abstraction
from linna.utils import load_model, load_tf_network, get_accuracy
from linna.network import Network

from timeit import default_timer as timer
from typing import Literal


def parse_network(network_file: str) -> torch.nn.Sequential:
    if network_file.endswith(".tf"):
        model = load_tf_network(file=network_file)
    else:
        model, _ = load_model(path=network_file)
    return model


def get_dataset(dataset_name):
    if "fashionmnist" in dataset_name.lower() or "fmnist" in dataset_name.lower() or "f-mnist" in dataset_name.lower():
        DOWNLOAD = not exists("../datasets/FMNIST")
        transform = transforms.Compose([transforms.ToTensor()])
        trainset = datasets.FashionMNIST('../datasets/FMNIST/TRAINSET', download=DOWNLOAD, train=True,
                                         transform=transform)
        testset = datasets.FashionMNIST('../datasets/FMNIST/TESTSET', download=DOWNLOAD, train=False,
                                        transform=transform)
    elif "kmnist" in dataset_name.lower():
        DOWNLOAD = not exists("../datasets/KMNIST")
        transform = transforms.Compose([transforms.ToTensor()])
        trainset = datasets.KMNIST('data/KMNIST/TRAINSET', download=DOWNLOAD, train=True, transform=transform)
        testset = datasets.KMNIST('data/KMNIST/TESTSET', download=DOWNLOAD, train=False, transform=transform)
    elif "mnist" in dataset_name.lower():
        DOWNLOAD = not exists("../datasets/MNIST")
        transform = transforms.Compose([transforms.ToTensor()])
        trainset = datasets.MNIST('data/MNIST/TRAINSET', download=DOWNLOAD, train=True, transform=transform)
        testset = datasets.MNIST('data/MNIST/TESTSET', download=DOWNLOAD, train=False, transform=transform)
    elif "cifar" in dataset_name.lower():
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        DOWNLOAD = not exists("../datasets/CIFAR10")
        trainset = datasets.CIFAR10('data/CIFAR10/TRAINSET', download=DOWNLOAD, train=True, transform=transform)
        testset = datasets.CIFAR10('data/CIFAR10/TESTSET', download=DOWNLOAD, train=False, transform=transform)
    else:
        raise NotImplementedError("We currently support only MNIST, FashionMNIST, KMNIST, and CIFAR10!")
    return trainset, testset


def run_layers(torch_model: torch.nn.Sequential, dataset: str, reduction_rate: float,
        layers: list,
        coef_finder: Literal["l1", "l2", "kmeans", "clustering", "dummy"] = "l2",
        basisfinder: Literal["greedy", "greedy_pruning", "variance", "kmeans", "dbscan", "random"] = "variance",
        syntactic: bool = False) -> Abstraction:
    """
        Run LiNNA

        Parameters
        ----------
        torch_model: torch.nn.Sequential
            The network to be abstracted and refined
        dataset: str
            The name of the datset to be used. Note that we don't support all datasets yet.
        reduction_rate: float
            Rate of how much to reduce the network
        coef_finder: Literal["l1", "l2", "kmeans", "clustering", "dummy"]
            How to find the coefficients for the linear combinations. (L1 requires Gurobi)
        basisfinder: Literal["greedy", "greedy_pruning", "variance", "kmeans", "dbscan", "random"]
            How to find the basis
        syntactic: bool
            whether to use syntactic information
        """
    network = Network(torch_model=torch_model)
    trainset, testset = get_dataset(dataset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    accuracy = get_accuracy(testloader, network.torch_model)

    print(f"### Start abstraction ###")
    print(f" Abstracting a {dataset}-network")
    print(f" using {basisfinder} and {coef_finder}")
    print(f" with test-accuracy {accuracy * 100:4.2f}%")
    print("Network:")
    print(network.torch_model)
    print("")
    start = timer()
    abstraction = Abstraction(
        network=network,
        basis_finder=basisfinder,
        coef_finder=coef_finder,
        loader=trainloader,
        size=1000,
        syntactic=syntactic
    )
    print(f" - Get the bases")
    for layer_idx in layers:
        abstraction.determine_basis_rr(layer_idx, reduction_rate=reduction_rate)
    print(f"   -> done")
    print(f" - Get the coefficients")
    for layer_idx in layers:
        abstraction.abstract(layer_idx)
    print(f"   -> done")
    end = timer()
    accuracy = get_accuracy(testloader, abstraction.network.torch_model)

    print(f"\n### Abstraction finished ###")
    print(
        f"Reduction rate: {abstraction.get_reduction_rate():4.2f}\nDuration: {end - start:10.2f}s\nTest-accuracy: {accuracy * 100:4.2f}%")
    print("Abstraction:")
    print(abstraction.network.torch_model)

    return abstraction

def run(torch_model: torch.nn.Sequential, dataset: str, reduction_rate: float,
        coef_finder: Literal["l1", "l2", "kmeans", "clustering", "dummy"] = "l2",
        basisfinder: Literal["greedy", "greedy_pruning", "variance", "kmeans", "dbscan", "random"] = "variance",
        syntactic: bool = False) -> Abstraction:
    """
    Run LiNNA

    Parameters
    ----------
    torch_model: torch.nn.Sequential
        The network to be abstracted and refined
    dataset: str
        The name of the datset to be used. Note that we don't support all datasets yet.
    reduction_rate: float
        Rate of how much to reduce the network
    coef_finder: Literal["l1", "l2", "kmeans", "clustering", "dummy"]
        How to find the coefficients for the linear combinations. (L1 requires Gurobi)
    basisfinder: Literal["greedy", "greedy_pruning", "variance", "kmeans", "dbscan", "random"]
        How to find the basis
    syntactic: bool
        whether to use syntactic information
    """
    network = Network(torch_model=torch_model)
    trainset, testset = get_dataset(dataset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    accuracy = get_accuracy(testloader, network.torch_model)

    print(f"### Start abstraction ###")
    print(f" Abstracting a {dataset}-network")
    print(f" using {basisfinder} and {coef_finder}")
    print(f" with test-accuracy {accuracy * 100:4.2f}%")
    print("Network:")
    print(network.torch_model)
    print("")
    start = timer()
    abstraction = Abstraction(
        network=network,
        basis_finder=basisfinder,
        coef_finder=coef_finder,
        loader=trainloader,
        size=1000,
        syntactic=syntactic
    )
    print(f" - Get the bases")
    abstraction.determine_bases(reduction_rate=reduction_rate)
    print(f"   -> done")
    print(f" - Get the coefficients")
    abstraction.abstract_all()
    print(f"   -> done")
    end = timer()
    accuracy = get_accuracy(testloader, abstraction.network.torch_model)

    print(f"\n### Abstraction finished ###")
    print(
        f"Reduction rate: {abstraction.get_reduction_rate():4.2f}\nDuration: {end - start:10.2f}s\nTest-accuracy: {accuracy * 100:4.2f}%")
    print("Abstraction:")
    print(abstraction.network.torch_model)

    return abstraction


def main():
    def is_valid_file(parser, arg):
        if not exists(arg):
            parser.error(f"The file {arg} does not exist.")
        else:
            return arg

    def is_valid_dataset(parser, arg):
        if arg.lower() in ["mnist", "kmnist", "k-mnist", "fmnist", "f-mnist", "fashionmnist", "cifar10", "cifar"]:
            return arg
        else:
            parser.error(f"We don't support the {arg}-dataset. Please use mnist,kmnist,fashionmnist or cifar10.")

    parser = argparse.ArgumentParser(prog="linna",
                                     description="How to use LiNNA (Linear Neural Network Abstraction).")

    parser.add_argument("-n", "--network",
                        type=(lambda x: is_valid_file(parser, x)),
                        help="This takes in the network that should be abstracted. The file format can be tensorflow "
                             "(.tf) or pytorch.sequential (created by torch.save)",
                        required=True)

    parser.add_argument("-rr", "--reduction-rate",
                        type=float,
                        required=True,
                        help="The reduction rate of how far to abstract the network.")

    parser.add_argument("-b", "--basisfinder",
                        type=str,
                        help="This takes in the variant how to find a basis.",
                        choices=["greedy", "greedy_pruning", "variance", "kmeans", "dbscan", "random"],
                        default="variance")

    parser.add_argument("-c", "--coefficentfinder",
                        type=str,
                        help="This takes in the variant how to find the coefficients. Only to be set for greedy, greedy-pruning or variance.",
                        choices=["l1", "l2"],
                        default="l2")

    parser.add_argument("-s", "--syntactic",
                        help="Switch to syntactic abstraction.",
                        action='store_true')

    parser.add_argument("-d", "--dataset",
                        type=(lambda x: is_valid_dataset(parser, x)),
                        help="Provide the dataset if you want semantic abstraction. It must be the same as the networkw as trained on. Better make sure that we use the same preprocessing!",
                        required=True)

    parser.add_argument("-st", "--store",
                        type=str,
                        help="Save the abstraction under this path.")

    args = parser.parse_args()
    torch_model = parse_network(args.network)

    if args.basisfinder in ["kmeans, dbscan"]:
        coef_finder = "clustering"
    else:
        coef_finder = args.coefficentfinder

    abstraction = run(torch_model, args.dataset, args.reduction_rate, coef_finder, args.basisfinder, args.syntactic)

    if args.store is not None:
        torch.save(abstraction.network.torch_model, args.store)
        print(f"The abstraction was saved at {args.store}.")

    return abstraction.network.torch_model


if __name__ == "__main__":
    main()
