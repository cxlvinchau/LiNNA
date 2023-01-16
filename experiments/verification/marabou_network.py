import sys

from experiments.playground.marabou_utils import print_equation, print_relu_constr
from linna.network import Network

sys.path.append('/home/calvin/Repositories/Marabou')

from maraboupy import MarabouCore
from maraboupy import Marabou
from maraboupy.MarabouNetwork import MarabouNetwork

import numpy as np

options = Marabou.createOptions(verbosity=0)


def linna_to_marabou_network(network: Network, x=None, delta=0, target=None) -> None:
    marabou_network = MarabouNetwork()
    # Setup variables
    variables = []
    # Input variables
    input_variables = []
    for _ in list(range(network.layers[0].get_weight().shape[1])):
        input_variables.append(marabou_network.getNewVariable())
    variables += input_variables

    # Setup equations
    output_variables = []

    pre_activation_variables = input_variables
    post_activation_variables = input_variables
    for idx, layer in enumerate(network.layers):
        weight = layer.get_weight()
        bias = layer.get_bias()
        new_post_activation_variables = []
        new_pre_activation_variables = []
        for neuron in layer.neurons:
            var = marabou_network.getNewVariable()
            # If neuron is in the basis
            if neuron not in layer.neuron_to_lower_bound or True:
                coeffs = []
                new_pre_activation_variables.append(var)
                for neuron_idx, in_var in enumerate(post_activation_variables):
                    coeffs.append(weight.cpu().detach().numpy().astype("float64")[neuron][neuron_idx])
                coeffs.append(-1)
                if idx < len(network.layers) - 1:
                    relu_var = marabou_network.getNewVariable()
                    marabou_network.setLowerBound(relu_var, 0)
                    new_post_activation_variables.append(relu_var)
                    marabou_network.addRelu(var, relu_var)
                else:
                    output_variables.append(var)
                marabou_network.addEquality(vars=post_activation_variables + [var],
                                            coeffs=coeffs,
                                            scalar=-bias.cpu().detach().numpy().astype("float64")[neuron])
            else:
                new_pre_activation_variables.append(var)
                new_post_activation_variables.append(var)

            # Sanity check - All post activation variables are greater equals zero
            # for var in post_activation_variables:
            #     equation = MarabouCore.Equation(MarabouCore.Equation.GE)
            #     equation.addAddend(1, var)
            #     equation.setScalar(0)
            #     inputQuery.addEquation(equation)

        if idx < len(network.layers) - 1:
            old_post_activations = post_activation_variables
            post_activation_variables = new_post_activation_variables
            pre_activation_variables = new_pre_activation_variables
            if idx == 0:
                debug_postactivations = post_activation_variables
            if idx == 1:
                debug_preactivations = pre_activation_variables

    print(len(input_variables))
    for input_var in input_variables:
        marabou_network.setLowerBound(input_var, x[input_var] - delta)
        marabou_network.setUpperBound(input_var, x[input_var] + delta)

    print(f"NUMBER of Variables: {len(variables)}")

    if target is not None:
        print(f"TARGET: {target}")
        target_var = output_variables[target]
        for idx, output_var in enumerate(output_variables):
            if idx != target:
                marabou_network.addInequality(vars=[output_var, target_var], coeffs=[-1, 1], scalar=0)

    var_sum = 0
    for equation in marabou_network.equList:
        var_sum += len(equation.addendList)
        print_equation(equation)

    for relu in marabou_network.reluList:
        print_relu_constr(relu)

    print(len(marabou_network.lowerBounds))
    print(len(marabou_network.upperBounds))
    marabou_network.outputVars = output_variables
    marabou_network.inputVars = [np.expand_dims(input_variables, 0)]
    marabou_network.outputVars = [np.expand_dims(output_variables, 0)]
    print(marabou_network.inputVars)
    print(marabou_network.outputVars)
    print(var_sum)

    status, result, stats = marabou_network.solve(options=options)
    print("===========================")
    print(f"Total time: {stats.getTotalTimeInMicro()} ms")
    print(status.upper())
    if status == "sat":
        for var in output_variables:
            print(result[var])

    # weight = network.layers[1].get_weight().cpu().detach().numpy()
    # bias = network.layers[1].get_bias().cpu().detach().numpy()
    # basis = network.layers[1].basis
    # print("\n\n")
    # for neuron, debug_var, relu_var in variable_pairs:
    #     if neuron != 41 and False:
    #         continue
    #     coef = network.layers[1].neuron_to_lower_bound[neuron]
    #     print(f"NEURON {neuron}")
    #     print(debug_postactivations)
    #     print(debug_var, relu_var)
    #     print(f"{result[debug_var]} >= {result[relu_var]}")
    #     value = 0
    #     for idx, tmp_neuron in enumerate(debug_postactivations):
    #         value += weight[neuron][idx] * result[tmp_neuron]
    #     value += bias[neuron]
    #     print(value)
    #     print(np.all(np.matmul(weight.T[:, basis], coef) <= weight.T[:, neuron]))
    #     print(np.all(np.matmul(bias[basis], coef) <= bias[neuron]))
    #     print()
    #     print()
