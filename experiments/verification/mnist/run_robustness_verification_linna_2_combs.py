import sys
sys.path.append('/home/calvin/Repositories/Marabou')
sys.path.append('/home/calvin/Repositories/LiNNA')

from linna.basis_finder import _BasisFinder, PosBasisFinder
from linna.network import Network
import torch
import sys
from timeit import default_timer as timer
import pandas as pd
import time

start = timer()

from linna.utils import load_tf_network, is_real_cex
from linna.verification.bounds import lp_upper_bound
from linna.verification.marabou_utils import evaluate_local_robustness, get_input_query
import numpy as np

from torchvision import datasets, transforms

from maraboupy import MarabouCore
from maraboupy import Marabou


def run_linna(network: Network, x: torch.Tensor, target_cls: int, delta: float, bf: _BasisFinder, reduction_rate,
              marabou_options):
    def abstract(layer_idx, basis_size):
        layer = network.layers[layer_idx]
        weight = layer.get_weight().cpu().detach().numpy()
        bias = layer.get_bias().cpu().detach().numpy()
        layer.basis = bf.find_basis(layer_idx=layer_idx, basis_size=basis_size)
        for neuron in layer.neurons:
            if neuron not in layer.basis:
                layer.neuron_to_lower_bound[neuron] = weight[neuron, :], bias[neuron]
                try:
                    linear_term, affine_term = lp_upper_bound(network=network,
                                                              layer_idx=layer_idx,
                                                              neuron=neuron,
                                                              non_negative=True,
                                                              semantic=False
                                                              )
                    layer.neuron_to_upper_bound[neuron] = linear_term
                    layer.neuron_to_upper_bound_affine_term[neuron] = affine_term
                except:
                    print(f"Could not find combination for {neuron}")

    for layer_idx, (basis_size, layer) in enumerate(network.layers[:-1]):
        abstract(layer_idx, int(len(layer.neurons) * (1 - reduction_rate)))

    cex, stats, max_class = evaluate_local_robustness(network=network,
                                                      x=x.cpu().detach().numpy(),
                                                      delta=delta,
                                                      target_cls=target_cls,
                                                      marabou_options=marabou_options,
                                                      bounds_type="syntactic")

    return cex, stats, max_class



if __name__ == "__main__":
    # Experiment parameters
    DELTAS = [0.02, 0.05]
    MARABOU_TIMEOUT = 10 * 60 # seconds
    NETWORKS = ["MNIST_3x100"]
    BASIS_SIZES = [None, 95, 90]

    # Load dataset
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST('../../datasets/MNIST/TRAINSET', download=False, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=60, shuffle=False)

    # Get first batch
    X, Y = next(iter(trainloader))

    # DataFrame rows
    rows = []

    for network in NETWORKS:
        for idx, (x, y) in enumerate(zip(X, Y)):
            target_cls = y.item()
            for delta in DELTAS:
                print(51 * "=")
                print(f"Configuration: {network}, img_idx: {idx}, delta: {delta}".upper())
                print(51 * "=")
                print("")

                # Information on configuration
                row = dict()
                row["network"] = network
                row["image_idx"] = idx
                row["target_cls"] = target_cls
                row["delta"] = delta

                # Run verification with LiNNA
                # ===========================
                try:
                    sequential = load_tf_network(file=f"../../networks/{network}.tf")
                    linna_net = Network(sequential)

                    # Start of computation
                    start = timer()
                    bf = PosBasisFinder(network=linna_net,
                                        io_dict={
                                            idx: layer.get_weight().cpu().detach().numpy().T for idx, layer in
                                            enumerate(linna_net.layers)
                                        })
                    marabou_options = Marabou.createOptions(verbosity=0,
                                                            timeoutInSeconds=MARABOU_TIMEOUT)
                    cex, stats, max_class = run_linna(network=linna_net,
                                                      x=x.view(-1, 784)[0],
                                                      target_cls=target_cls,
                                                      delta=delta,
                                                      bf=bf,
                                                      marabou_options=marabou_options
                                                      )
                    end = timer()
                    # End of computation

                    row["linna_time (seconds)"] = end - start
                    row["linna_timeout"] = stats == "timeout"
                    row["linna_is_real_cex"] = None
                    row["linna_cex"] = None
                    if cex is not None:
                        row["linna_result"] = "sat"
                        row["linna_cex"] = [cex[i] for i in range(784)]
                        row["linna_is_real_cex"] = is_real_cex(network=linna_net,
                                                               cex=torch.Tensor([cex[i] for i in range(784)]),
                                                               target_cls=target_cls)
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
                df.to_csv(f"robust_experiment_data/{timestr}.csv")