from linna.basis_finder import _BasisFinder, PosBasisFinder
from linna.network import Network
import torch
import sys
from timeit import default_timer as timer
import pandas as pd
import time

start = timer()

from linna.utils import load_tf_network
from linna.verification.bounds import lp_upper_bound
from linna.verification.marabou_utils import evaluate_local_robustness, get_input_query
import numpy as np

from torchvision import datasets, transforms

sys.path.append('/home/calvin/Repositories/Marabou')
from maraboupy import MarabouCore
from maraboupy import Marabou


def run_linna(network: Network, x: torch.Tensor, target_cls: int, delta: float, bf: _BasisFinder, basis_sizes,
              marabou_options):
    assert len(basis_sizes) == len(network.layers) - 1

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

    for layer_idx, (basis_size, layer) in enumerate(zip(basis_sizes, network.layers[:-1])):
        if basis_size is not None:
            abstract(layer_idx, basis_size)

    cex, stats, max_class = evaluate_local_robustness(network=network,
                                                      x=x.cpu().detach().numpy(),
                                                      delta=delta,
                                                      target_cls=target_cls,
                                                      marabou_options=marabou_options)

    return cex, stats, max_class


def is_real_cex(network: Network, cex: torch.Tensor, target_cls: int):
    out = network.forward(cex).cpu().detach().numpy()
    max_classes = np.argwhere(out == np.max(out)).reshape(-1).tolist()
    return target_cls not in max_classes or len(max_classes) > 1


if __name__ == "__main__":
    # Experiment parameters
    DELTAS = [0.02, 0.05]
    MARABOU_TIMEOUT = 10 * 60  # seconds
    NETWORKS = ["MNIST_3x100"]
    BASIS_SIZES = [None, 95, 90]

    # Load dataset
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST('../../datasets/MNIST/TRAINSET', download=False, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=False)

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
                                                      basis_sizes=BASIS_SIZES,
                                                      marabou_options=marabou_options
                                                      )
                    end = timer()
                    # End of computation

                    row["linna_time (seconds)"] = end - start
                    row["linna_timeout"] = stats == "timeout"
                    row["linna_is_real_cex"] = None
                    if cex is not None:
                        row["linna_result"] = "sat"
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



                # Run verification with Marabou only
                # ==================================
                try:
                    # Export network
                    sequential = load_tf_network(file=f"../../networks/{network}.tf")
                    linna_net = Network(sequential)
                    torch.onnx.export(linna_net.torch_model, x.view(-1, 784), "tmp/model.onnx")

                    marabou_net = Marabou.read_onnx("tmp/model.onnx")

                    target_cls = y.item()

                    marabou_options = Marabou.createOptions(verbosity=0,
                                                            timeoutInSeconds=MARABOU_TIMEOUT)

                    start = timer()
                    result, stats, max_class = marabou_net.evaluateLocalRobustness(
                        x.view(-1, 784)[0].cpu().detach().numpy(), delta, target_cls, options=marabou_options)
                    end = timer()

                    row["marabou_time (seconds)"] = end - start
                    row["marabou_timeout"] = stats.hasTimedOut()
                    if stats.hasTimedOut():
                        row["marabou_result"] = "timeout"
                    elif result is None or len(result) == 0:
                        row["marabou_result"] = "unsat"
                    else:
                        row["marabou_result"] = "sat"
                    row["marabou_error"] = False
                except:
                    row["marabou_error"] = True


                rows.append(pd.Series(row))

                print(pd.Series(row))

                timestr = time.strftime("%Y%m%d-%H%M%S")
                df = pd.DataFrame(rows)
                df.to_csv(f"robust_experiment_data/{timestr}.csv")