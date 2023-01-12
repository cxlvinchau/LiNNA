import sys

from linna.network import Network

sys.path.append('/home/calvin/Documents/tools/Marabou')

from maraboupy import MarabouCore
from maraboupy import Marabou

import numpy as np


def print_equation(equation):
    l = [f"{type(c)}*x{v}" for c, v in equation.addendList]

    if equation.EquationType == MarabouCore.Equation.EQ:
        eq_type = "="
    elif equation.EquationType == MarabouCore.Equation.LE:
        eq_type = "<="
    elif equation.EquationType == MarabouCore.Equation.LE:
        eq_type = "=<"

    print(f"{' + '.join(l)} {eq_type} {type(equation.scalar)}")


def print_relu_constr(t):
    print(f"x{t[1]} = ReLU({t[0]})")


options = Marabou.createOptions()


def linna_to_marabou(network: Network, x=None, delta=0, target=None) -> None:
    inputQuery = MarabouCore.InputQuery()
    # Setup variables
    variables = []
    # Input variables
    input_variables = list(range(network.layers[0].get_weight().shape[1]))
    variables += input_variables
    free_variables = []

    def get_new_variable():
        variables.append(len(variables))
        return len(variables) - 1

    # Setup equations
    output_variables = []
    debug_variables = []
    variable_pairs = []
    debug_preactivations = []
    debug_postactivations = []

    pre_activation_variables = input_variables
    post_activation_variables = input_variables
    old_post_activations = None
    for idx, layer in enumerate(network.layers):
        weight = layer.get_weight()
        bias = layer.get_bias()
        new_post_activation_variables = []
        new_pre_activation_variables = []
        for neuron in layer.neurons:
            var = get_new_variable()
            # If neuron is in the basis
            if neuron not in layer.neuron_to_lower_bound:
                equation = MarabouCore.Equation()
                new_pre_activation_variables.append(var)
                for neuron_idx, in_var in enumerate(post_activation_variables):
                    equation.addAddend(weight[neuron][neuron_idx], in_var)
                equation.addAddend(-1, var)
                equation.setScalar(-bias[neuron])
                if idx < len(network.layers) - 1:
                    relu_var = get_new_variable()
                    new_post_activation_variables.append(relu_var)
                    MarabouCore.addReluConstraint(inputQuery, var, relu_var)
                else:
                    output_variables.append(var)
                inputQuery.addEquation(equation)
            else:
                relu_var = get_new_variable()
                new_pre_activation_variables.append(var)
                new_post_activation_variables.append(relu_var)
                equation = MarabouCore.Equation()
                equation.addAddend(-1, var)
                equation.addAddend(1, relu_var)
                equation.setScalar(0)
                inputQuery.addEquation(equation)

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

        # Test bounds
        for neuron, relu_var in zip(layer.neurons, post_activation_variables):
            if neuron in layer.neuron_to_lower_bound and idx < len(network.layers) - 1:
                # Lower bound
                equation = MarabouCore.Equation(MarabouCore.Equation.GE)
                equation.setScalar(0)
                equation.addAddend(1, post_activation_variables[neuron])
                # Greater equals zero due to ReLU
                equation = MarabouCore.Equation(MarabouCore.Equation.LE)
                for basis_idx, basis_neuron in enumerate(layer.basis):
                    equation.addAddend(layer.neuron_to_lower_bound[neuron][basis_idx],
                                       pre_activation_variables[basis_neuron])
                equation.addAddend(-1, post_activation_variables[neuron])
                equation.setScalar(0)
                inputQuery.addEquation(equation)

                # Upper bound
                equation = MarabouCore.Equation(MarabouCore.Equation.GE)
                for basis_idx, basis_neuron in enumerate(layer.basis):
                    equation.addAddend(layer.neuron_to_upper_bound[neuron][basis_idx],
                                       post_activation_variables[basis_neuron])

                for neuron_idx, in_var in enumerate(old_post_activations):
                    equation.addAddend(layer.neuron_to_upper_bound_term[neuron][neuron_idx], in_var)

                equation.addAddend(-1, post_activation_variables[neuron])
                equation.setScalar(-layer.neuron_to_upper_bound_term[neuron][-1])
                inputQuery.addEquation(equation)

                free_variables.append(post_activation_variables[neuron])

    inputQuery.setNumberOfVariables(len(variables))

    print(len(input_variables))
    for input_var in input_variables:
        inputQuery.setLowerBound(input_var, x[input_var] - delta)
        inputQuery.setUpperBound(input_var, x[input_var] + delta)

    i = 0
    for var in input_variables + free_variables:
        inputQuery.markInputVariable(var, i)
        i += 1

    i = 0
    for output_var in output_variables:
        inputQuery.markOutputVariable(output_var, i)
        i += 1

    print(f"NUMBER of Variables: {len(variables)}")
    print(f"Out vars: {len(output_variables)}")

    if target is not None:
        print(f"TARGET: {target}")
        target_var = output_variables[target]
        for idx, output_var in enumerate(output_variables):
            if idx != target:
                equation = MarabouCore.Equation(MarabouCore.Equation.LE)
                equation.addAddend(-1, output_var)
                equation.addAddend(1, target_var)
                equation.setScalar(0)
                inputQuery.addEquation(equation)

    status, result, stats = MarabouCore.solve(inputQuery, options)
    print("===========================")
    print(f"Total time: {stats.getTotalTimeInMicro()} ms")
    print(status.upper())
    if status == "sat":
        for idx, var in enumerate(output_variables):
            print(f"Out {idx}: {result[var]}")

        return np.array([result[var] for var in input_variables])

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
