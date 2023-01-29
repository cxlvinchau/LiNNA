import sys
sys.path.append('/home/calvin/Repositories/Marabou')
sys.path.append('/home/calvin/Repositories/LiNNA')

from linna.coef_finder import _CoefFinder, L1CoefFinder, L2CoefFinder
import time
import pandas as pd
from timeit import default_timer as timer
import torch
from linna.network import Network
from linna.basis_finder import _BasisFinder, PosBasisFinder, VarianceBasisFinder
from linna.utils import load_tf_network
from linna.verification.bounds import lp_upper_bound
from linna.verification.marabou_utils import evaluate_local_robustness, get_input_query
import numpy as np
from torchvision import datasets, transforms
from maraboupy import MarabouCore
from maraboupy import Marabou



start = timer()


def run_linna_semantic(network: Network, x: torch.Tensor, target_cls: int, delta: float, bf: _BasisFinder, basis_sizes, cf: _CoefFinder,
                       lb_epsilon: float, ub_epsilon: float, marabou_options):
    assert len(basis_sizes) == len(network.layers) - 1

    def abstract(layer_idx, basis_size):
        layer = network.layers[layer_idx]
        weight = layer.get_weight().cpu().detach().numpy()
        bias = layer.get_bias().cpu().detach().numpy()
        layer.basis = bf.find_basis(layer_idx=layer_idx, basis_size=basis_size)
        for neuron in layer.neurons:
            if neuron not in layer.basis:
                layer.neuron_to_coef[neuron] = cf.find_coefficients(
                    layer_idx=layer_idx, neuron=neuron)

    for layer_idx, (basis_size, layer) in enumerate(zip(basis_sizes, network.layers[:-1])):
        if basis_size is not None:
            abstract(layer_idx, basis_size)

    cex, stats, max_class = evaluate_local_robustness(network=network,
                                                      x=x.cpu().detach().numpy(),
                                                      delta=delta,
                                                      target_cls=target_cls,
                                                      marabou_options=marabou_options,
                                                      bounds_type="semantic",
                                                      params_dict={"lb_epsilon": lb_epsilon,
                                                                   "ub_epsilon": ub_epsilon})

    return cex, stats, max_class


def is_real_cex(network: Network, cex: torch.Tensor, target_cls: int):
    out = network.forward(cex).cpu().detach().numpy()
    max_classes = np.argwhere(out == np.max(out)).reshape(-1).tolist()
    return target_cls not in max_classes or len(max_classes) > 1


def compute_io_dict(network: Network, loader):
    # Compute IO dict
    io_dict = dict()
    for layer in range(len(network.layers)):
        io_dict[layer] = network.get_io_matrix(
            loader=loader, layer_idx=layer, size=1000)
    return io_dict


if __name__ == "__main__":
    # Experiment parameters
    DELTAS = [0.02, 0.05]
    MARABOU_TIMEOUT = 10 * 60  # seconds
    NETWORKS = ["MNIST_3x100"]
    BASIS_SIZES = [None, 95, 90]
    BOUNDS = [(3, 3), (0, 0)]

    # Load dataset
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST(
        '../../datasets/MNIST/TRAINSET', download=False, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=50, shuffle=False)

    # Get first batch
    X, Y = next(iter(trainloader))

    # DataFrame rows
    rows = []

    for network in NETWORKS:
        for idx, (x, y) in enumerate(zip(X, Y)):
            target_cls = y.item()
            for delta in DELTAS:
                for lb_epsilon, ub_epsilon in BOUNDS:
                    print(80 * "=")
                    print(
                        f"Configuration: {network}, img_idx: {idx}, delta: {delta}, bounds: {lb_epsilon, ub_epsilon}".upper())
                    print(80 * "=")
                    print("")

                    # Information on configuration
                    row = dict()
                    row["network"] = network
                    row["image_idx"] = idx
                    row["target_cls"] = target_cls
                    row["delta"] = delta
                    row["bounds"] = (lb_epsilon, ub_epsilon)

                    try:
                        # Run verification with LiNNA
                        # ===========================
                        sequential = load_tf_network(
                            file=f"../../networks/{network}.tf")
                        linna_net = Network(sequential)

                        # Start of computation
                        start = timer()
                        io_transform = transforms.Compose([transforms.ToTensor()])
                        io_set = datasets.MNIST('../../datasets/MNIST/TRAINSET', download=False, train=True, transform=io_transform)
                        io_loader = torch.utils.data.DataLoader(
                            io_set, batch_size=64, shuffle=True)
                        io_dict = compute_io_dict(
                            network=linna_net, loader=io_loader)
                        bf = VarianceBasisFinder(network=linna_net,
                                                io_dict=io_dict)
                        cf = L1CoefFinder(network=linna_net,
                                        io_dict=io_dict)
                        marabou_options = Marabou.createOptions(verbosity=0,
                                                                timeoutInSeconds=MARABOU_TIMEOUT)
                        cex, stats, max_class = run_linna_semantic(network=linna_net, x=x.view(-1, 784)[0],
                                                                target_cls=target_cls,
                                                                delta=delta,
                                                                bf=bf,
                                                                cf=cf,
                                                                basis_sizes=BASIS_SIZES,
                                                                marabou_options=marabou_options,
                                                                lb_epsilon=lb_epsilon,
                                                                ub_epsilon=ub_epsilon
                                                                )
                        end = timer()
                        # End of computation

                        row["linna_time (seconds)"] = end - start
                        row["linna_timeout"] = stats == "timeout"
                        row["linna_is_real_cex"] = None
                        row["linna_cex"] = None
                        if cex is not None:
                            row["linna_result"] = "sat"
                            row["linna_is_real_cex"] = is_real_cex(network=linna_net,
                                                                cex=torch.Tensor(
                                                                    [cex[i] for i in range(784)]),
                                                                target_cls=target_cls)
                            row["linna_cex"] = [cex[i] for i in range(784)]
                        elif stats == "timeout":
                            row["linna_result"] = "timeout"
                        else:
                            row["linna_result"] = "unsat"

                        row["linna_error"] = False
                    except:
                        row["linna_error"] = True

                    rows.append(pd.Series(row))

                    print(pd.Series(row))

                    timestr = time.strftime("%Y%m%d-%H%M%S")
                    df = pd.DataFrame(rows)
                    df.to_csv(f"robust_experiment_data_semantic/{timestr}.csv")