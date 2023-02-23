import argparse
from os.path import exists, isfile

import torch
from torchvision import datasets, transforms
from linna.abstraction import Abstraction
from linna.utils import load_model, load_tf_network, get_accuracy
from linna.network import Network

from timeit import default_timer as timer


def parse_network(network_file: str) -> torch.nn.Sequential:
    if network_file.endswith(".tf"):
        model = load_tf_network(file=network_file)
    else:
        model, _ = load_model(path=network_file)
    return model


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

    network = Network(torch_model=parse_network(args.network))
    if args.basisfinder in ["kmeans, dbscan"]:
        coef_finder = "clustering"
    else:
        coef_finder = args.coefficentfinder

    trainset, testset = get_dataset(args.dataset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    accuracy = get_accuracy(testloader, network.torch_model)

    print(f"### Start abstraction ###")
    print(f" Abstracting a {args.dataset}-network")
    print(f" using {args.basisfinder} and {coef_finder}")
    print(f" with test-accuracy {accuracy*100:4.2f}%")
    print("Network:")
    print(network.torch_model)
    print("")
    start = timer()
    abstraction = Abstraction(
        network=network,
        basis_finder=args.basisfinder,
        coef_finder=coef_finder,
        loader=trainloader,
        size=1000,
        syntactic=args.syntactic
    )
    print(f" - Get the bases")
    abstraction.determine_bases(reduction_rate=args.reduction_rate)
    print(f"   -> done")
    print(f" - Get the coefficients")
    abstraction.abstract_all()
    print(f"   -> done")
    end = timer()
    accuracy = get_accuracy(testloader, abstraction.network.torch_model)

    print(f"\n### Abstraction finished ###")
    print(f"Reduction rate: {abstraction.get_reduction_rate():4.2f}\nDuration: {end-start:10.2f}s\nTest-accuracy: {accuracy*100:4.2f}%")
    print("Abstraction:")
    print(abstraction.network.torch_model)

    if args.store is not None:
        torch.save(abstraction.network.torch_model, args.store)
        print(f"The abstraction was saved at {args.store}.")



if __name__ == "__main__":
    main()
