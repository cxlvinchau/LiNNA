from linna.abstraction import Abstraction
from linna.basis_finder import VarianceBasisFinder
from linna.network import Network
from torch.utils.data.dataset import Dataset
from linna.utils import get_accuracy, load_tf_network
import sys
from torchvision import datasets, transforms
import torch
import numpy as np

from linna.verification.bounds import lp_upper_bound, lp_upper_bound_semantic

sys.path.append('/home/calvin/Documents/tools/Marabou')
from maraboupy import Marabou
from maraboupy import MarabouCore

# Load MNIST dataset
# Set to true to download MNIST data set
DOWNLOAD = False
NETWORK = "MNIST_3x100.tf"
# Get training data
transform = transforms.Compose([transforms.ToTensor()])
# Get data
trainset = datasets.MNIST('../../datasets/MNIST/TRAINSET', download=DOWNLOAD, train=True, transform=transform)
testset = datasets.MNIST('../../datasets/MNIST/TESTSET', download=DOWNLOAD, train=False, transform=transform)

transform = transforms.Compose([transforms.ToTensor()])
# Create train and test loader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

LAYER_IDX = 1

OPTIONS = Marabou.createOptions(verbosity=0)

# Load trained neural network
sequential = load_tf_network(file=f"../../networks/{NETWORK}")
network = Network(torch_model=sequential)

# Compute IO matrices
io_dict = dict()
for layer in range(len(network.layers)):
    io_dict[layer] = network.get_io_matrix(loader=trainloader, layer_idx=layer, size=10000)
bf = VarianceBasisFinder(network=network,
                         io_dict=io_dict)


# bf = VarianceBasisFinder(network=network,
#                          io_dict={
#                              idx: layer.get_weight().cpu().detach().numpy().T for idx, layer in
#                              enumerate(network.layers)
#                          })
#
network.layers[LAYER_IDX].basis = bf.find_basis(layer_idx=LAYER_IDX, basis_size=70)


def check_lin_comb(lin_comb, lin_comb2, neuron):
    variables = []
    input_variables = []
    output_variables = []
    weights = network.layers[LAYER_IDX].get_weight().cpu().detach().numpy()
    bias = network.layers[LAYER_IDX].get_bias().cpu().detach().numpy()
    basis = network.layers[LAYER_IDX].basis

    ipq = MarabouCore.InputQuery()

    def get_var():
        variables.append(len(variables))
        return len(variables) - 1

    # Generate input variables
    for _ in range(weights.shape[0]):
        input_var = get_var()
        input_variables.append(input_var)

    # Basis variables
    post_activations = []
    neuron_var = None
    for basis_neuron in basis + [neuron]:
        basis_var = get_var()
        equation = MarabouCore.Equation(MarabouCore.Equation.EQ)
        for input_var in input_variables:
            equation.addAddend(weights[basis_neuron][input_var], input_var)
        equation.addAddend(-1, basis_var)
        equation.setScalar(bias[basis_neuron])
        ipq.addEquation(equation)
        relu_var = get_var()
        post_activations.append(relu_var)
        MarabouCore.addReluConstraint(ipq, basis_var, relu_var)
        if basis_neuron == neuron:
            neuron_var = relu_var

    # Verify upper bound holds
    lin_comb_var = get_var()
    equation = MarabouCore.Equation(MarabouCore.Equation.EQ)
    for idx, basis_var in enumerate(post_activations[:-1]):
        equation.addAddend(lin_comb[idx], basis_var)
    equation.addAddend(-1, lin_comb_var)
    equation.setScalar(-np.max(lin_comb2[-1]))
    ipq.addEquation(equation)

    equation = MarabouCore.Equation(MarabouCore.Equation.LE)
    equation.addAddend(1, lin_comb_var)
    equation.addAddend(-1, neuron_var)
    equation.setScalar(-1)
    ipq.addEquation(equation)

    ipq.setNumberOfVariables(len(variables))

    for input_var in input_variables:
        ipq.setLowerBound(input_var, 0)
        ipq.setUpperBound(input_var, 1)

    exitCode, vals, stats = MarabouCore.solve(ipq, options=OPTIONS)

    print(exitCode)
    if exitCode == "sat":
        for input_var in input_variables:
            print(f"{input_var}: {vals[input_var]}")
        print(f"Lin comb = {vals[lin_comb_var]}")
        print(f"Neuron = {vals[neuron_var]}")
        print(vals[lin_comb_var] >= vals[post_activations[-1]])


for neuron in network.layers[LAYER_IDX].neurons:
    if neuron not in network.layers[LAYER_IDX].basis:
        x, y = lp_upper_bound_semantic(network, layer_idx=LAYER_IDX, neuron=neuron, matrix=io_dict[LAYER_IDX])
        check_lin_comb(x, y, neuron)
        sys.exit(0)
