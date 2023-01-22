import gurobipy as gp
from gurobipy import GRB
import numpy as np
from torchvision import datasets, transforms
import torch

# Load network
from linna.basis_finder import VarianceBasisFinder
from linna.network import Network
from linna.utils import load_tf_network

# LiNNA setup
sequential = load_tf_network(file="../../networks/MNIST_5x100.tf")
linna_net = Network(sequential)
LAYER_IDX = 1

# Input img
transform = transforms.Compose([transforms.ToTensor()])
trainset = datasets.MNIST('../../datasets/MNIST/TRAINSET', download=False, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False)

img_idx = 5

X, y = next(iter(trainloader))
x = X[img_idx].view(-1, 784)[0].cpu().detach().numpy()
target_cls = y[img_idx].item()

# Model parameters
weight = linna_net.layers[LAYER_IDX].get_weight().cpu().detach().numpy()
bias = linna_net.layers[LAYER_IDX].get_bias().cpu().detach().numpy()
n_incoming = weight.shape[1]
n_outgoing = weight.shape[0]
lb, ub = linna_net.propagate_interval(x - 0.01, x + 0.01, layer_idx=LAYER_IDX - 1)


def check_is_dominating(neuron, lb, ub):
    assert np.all(lb <= ub)
    # Incoming variables
    m = gp.Model()
    x = m.addMVar(n_incoming, lb=lb, ub=ub)
    post_activation = m.addVar()
    pre_activation = m.addVar(lb=-GRB.INFINITY)
    m.addConstr(pre_activation == weight[neuron, :] @ x + bias[neuron])
    m.addGenConstrMax(post_activation, [pre_activation, 0])
    for other in linna_net.layers[LAYER_IDX].neurons:
        if other != neuron:
            pre_activation_other = m.addVar(lb=-GRB.INFINITY)
            m.addConstr(pre_activation_other == weight[other, :] @ x + bias[other])
            post_activation_other = m.addVar()
            m.addGenConstrMax(post_activation_other, [pre_activation_other, 0])
            m.addConstr(post_activation >= post_activation_other + 0.00001)
    m.setObjective(np.ones(n_incoming) @ x, sense=GRB.MINIMIZE)
    m.optimize()
    return m.status == 2


def determine_max_val(neurons, lb, ub):
    m = gp.Model()
    x = m.addMVar(n_incoming, lb=lb, ub=ub)
    c = m.addVar(lb=-GRB.INFINITY)
    for neuron in neurons:
        pre_activation = m.addVar(lb=-GRB.INFINITY)
        m.addConstr(pre_activation == weight[neuron, :] @ x + bias[neuron])
        post_activation = m.addVar()
        m.addGenConstrMax(post_activation, [pre_activation, 0])
        m.addConstr(c >= post_activation)
    m.setObjective(c, sense=GRB.MINIMIZE)
    m.optimize()


not_dominating = []
dominating = []
for neuron in linna_net.layers[LAYER_IDX].neurons:
    if not check_is_dominating(neuron, lb, ub):
        not_dominating.append(neuron)
    else:
        dominating.append(neuron)


print(not_dominating)
print(dominating)
print(determine_max_val(not_dominating, lb, ub))
