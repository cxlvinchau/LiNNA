import sys

from linna.network import Network

sys.path.append('/home/calvin/Repositories/Marabou')

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


def get_input_query(network: Network, bounds_type="syntactic", params_dict=None):
    ipq = MarabouCore.InputQuery()
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

    pre_activation_variables = input_variables
    post_activation_variables = input_variables
    old_post_activations = None
    pairs = []
    for idx, layer in enumerate(network.layers):
        weight = layer.get_weight()
        bias = layer.get_bias()
        new_post_activation_variables = []
        new_pre_activation_variables = []
        for neuron in layer.neurons:
            var = get_new_variable()
            # Generate equation for neurons in the basis
            if layer.basis is None or neuron in layer.basis:
                # Equation for pre-activation
                equation = MarabouCore.Equation()
                new_pre_activation_variables.append(var)
                for neuron_idx, in_var in enumerate(post_activation_variables):
                    equation.addAddend(weight[neuron][neuron_idx], in_var)
                equation.addAddend(-1, var)
                equation.setScalar(-bias[neuron])
                # Apply ReLU constraint if the layer is not the last layer
                if idx < len(network.layers) - 1:
                    relu_var = get_new_variable()
                    new_post_activation_variables.append(relu_var)
                    MarabouCore.addReluConstraint(ipq, var, relu_var)
                else:
                    output_variables.append(var)
                ipq.addEquation(equation)
            else:  # Treat neurons not in the basis as free variables
                new_pre_activation_variables.append(var)
                new_post_activation_variables.append(var)
                free_variables.append(var)

        # Update pre- and post-activation variables
        if idx < len(network.layers) - 1:
            old_post_activations = post_activation_variables
            post_activation_variables = new_post_activation_variables
            pre_activation_variables = new_pre_activation_variables

        # Compute equations for bounds
        for neuron, relu_var in zip(layer.neurons, post_activation_variables):
            if (neuron in layer.neuron_to_upper_bound or neuron in layer.neuron_to_coef) and idx < len(network.layers) - 1:
                # Create bound variables
                lb_var = get_new_variable()
                ub_var = get_new_variable()

                # Syntactic overapproximation
                if bounds_type == "syntactic":
                    # Lower bound equation
                    equation = MarabouCore.Equation(MarabouCore.Equation.EQ)
                    for var_idx, in_var in enumerate(old_post_activations):
                        equation.addAddend(layer.neuron_to_lower_bound[neuron][0][var_idx], in_var)
                    equation.addAddend(-1, lb_var)
                    equation.setScalar(-1 * layer.neuron_to_lower_bound[neuron][1])
                    ipq.addEquation(equation)

                    # Upper bound equation
                    equation = MarabouCore.Equation(MarabouCore.Equation.EQ)
                    for basis_idx, basis_neuron in enumerate(layer.basis):
                        equation.addAddend(layer.neuron_to_upper_bound[neuron][basis_idx],
                                        post_activation_variables[basis_neuron])

                    for input_idx, input_var in enumerate(old_post_activations):
                        equation.addAddend(layer.neuron_to_upper_bound_affine_term[neuron][input_idx], input_var)

                    equation.setScalar(-layer.neuron_to_upper_bound_affine_term[neuron][-1])
                    equation.addAddend(-1, ub_var)
                    ipq.addEquation(equation)
                elif bounds_type == "semantic":
                    # Assert that information is available
                    assert params_dict is not None and "lb_epsilon" in params_dict and "ub_epsilon" in params_dict
                    assert params_dict["lb_epsilon"] >= 0 and params_dict["ub_epsilon"] >= 0
                    
                    # Lower bound equation
                    equation = MarabouCore.Equation(MarabouCore.Equation.EQ)
                    for basis_idx, basis_neuron in enumerate(layer.basis):
                        basis_var = post_activation_variables[basis_neuron]
                        equation.addAddend(layer.neuron_to_coef[neuron][basis_idx], basis_var)
                    equation.setScalar(params_dict["lb_epsilon"])
                    equation.addAddend(-1, lb_var)
                    ipq.addEquation(equation)
                    
                    # Upper bound equation
                    equation = MarabouCore.Equation(MarabouCore.Equation.EQ)
                    for basis_idx, basis_neuron in enumerate(layer.basis):
                        basis_var = post_activation_variables[basis_neuron]
                        equation.addAddend(layer.neuron_to_coef[neuron][basis_idx], basis_var)
                    equation.setScalar(-params_dict["ub_epsilon"])
                    equation.addAddend(-1, ub_var)
                    ipq.addEquation(equation)

                else:
                    raise ValueError("Unknown bounds type!")

                # Constraint the neurons not in the basis
                # LB
                equation = MarabouCore.Equation(MarabouCore.Equation.LE)
                equation.addAddend(1, lb_var)
                equation.addAddend(-1, post_activation_variables[neuron])
                equation.setScalar(0)
                ipq.addEquation(equation)

                # UB
                equation = MarabouCore.Equation(MarabouCore.Equation.GE)
                equation.addAddend(1, ub_var)
                equation.addAddend(-1, post_activation_variables[neuron])
                equation.setScalar(0)
                ipq.addEquation(equation)

                pairs.append((lb_var, post_activation_variables[neuron], ub_var))

    ipq.setNumberOfVariables(len(variables))

    i = 0
    for var in input_variables + free_variables:
        ipq.markInputVariable(var, i)
        if var in free_variables:  # Free variables correspond to neurons not in basis, value cannot be < 0
            ipq.setLowerBound(var, 0)
        i += 1

    i = 0
    for output_var in output_variables:
        ipq.markOutputVariable(output_var, i)
        i += 1

    return ipq, input_variables, output_variables, pairs


def evaluate_local_robustness(network: Network, x, delta, target_cls, marabou_options, bounds_type="syntactic", params_dict=None):
    ipq, input_variables, output_variables, pairs = get_input_query(network, bounds_type=bounds_type, params_dict=params_dict)

    for input_var in input_variables:
        ipq.setLowerBound(input_var, x[input_var] - delta)
        ipq.setUpperBound(input_var, x[input_var] + delta)

    target_var = output_variables[target_cls]

    for idx, output_var in enumerate(output_variables):
        if idx != target_cls:
            MarabouCore.addMaxConstraint(ipq, set([output_var, target_var]), output_var)
            exitCode, vals, stats = MarabouCore.solve(ipq, options=marabou_options)
            print(80 * "=")
            print(f"Query for cls {idx}")
            print(f"Total time: {stats.getTotalTimeInMicro()} microseconds")
            if exitCode == "sat":
                # for cls, var in enumerate(output_variables):
                #     print(f"Output {cls}: {vals[var]}")
                diff = 0
                for lb, var, ub in pairs:
                    # assert max(vals[lb], 0) <= vals[var] <= vals[ub]
                    diff += vals[ub] - max(vals[lb], 0)
                return vals, stats, idx
            elif exitCode.lower() == "timeout":
                return None, "timeout", None

            print(80 * "=" + "\n")

    return None, None, None
