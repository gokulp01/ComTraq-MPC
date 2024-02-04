import pulp
import numpy as np


def make_decision(
    current_step,
    current_budget,
    cost_per_communication,
    current_window_cost,
    x_variance,
    y_variance,
    variance_importance_factor,
    true_state,
    belief_state,
):
    # Define the ILP problem
    prob = pulp.LpProblem("SingleStepDecision", pulp.LpMinimize)

    # Decision variable for this step
    decision = pulp.LpVariable("communicate", 0, 1, cat="Binary")

    # Objective function
    total_variance = x_variance + y_variance
    # Separate the cost of communication and the variance part in the objective
    prob += (1 - decision) * current_window_cost + (decision) * (
        np.linalg.norm(np.array(true_state) - np.array(belief_state))
    )
    print(f"prob:{prob}")
    # Constraints
    # Budget constraint for this decision
    prob += decision * cost_per_communication <= current_budget

    # Solve the problem
    pulp_solver = pulp.PULP_CBC_CMD(msg=False)  # Suppress PuLP's output
    prob.solve(pulp_solver)

    # Return the decision
    if prob.status == pulp.LpStatusOptimal:
        return "communicate" if pulp.value(decision) == 1 else "no_communicate"
    else:
        return "No feasible solution found"


# Example usage
# current_step = 1
# current_budget = 100000
# cost_per_communication = 1
# current_window_cost = 2.5  # Example cost
# x_variance = 1.0  # Example x variance at current step
# y_variance = 1.5  # Example y variance at current step
# variance_importance_factor = 1  # Adjust this factor to control how much variance influences the decision

# # Make the decision for the current step
# decision = make_decision(current_step, current_budget, cost_per_communication, current_window_cost, x_variance, y_variance, variance_importance_factor)
# print("Decision at step", current_step, ":", decision)
