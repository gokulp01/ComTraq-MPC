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
    prob = pulp.LpProblem("SingleStepDecision", pulp.LpMinimize)

    decision = pulp.LpVariable("communicate", 0, 1, cat="Binary")

    total_variance = x_variance + y_variance
    prob += (1 - decision) * current_window_cost + (decision) * (
        np.linalg.norm(np.array(true_state) - np.array(belief_state))
    )
    print(f"prob:{prob}")
    prob += decision * cost_per_communication <= current_budget

    pulp_solver = pulp.PULP_CBC_CMD(msg=False)  # Suppress PuLP's output
    prob.solve(pulp_solver)

    if prob.status == pulp.LpStatusOptimal:
        return "communicate" if pulp.value(decision) == 1 else "no_communicate"
    else:
        return "No feasible solution found"


# current_step = 1
# current_budget = 100000
# cost_per_communication = 1
# current_window_cost = 2.5  # Example cost
# x_variance = 1.0  # Example x variance at current step
# y_variance = 1.5
# variance_importance_factor = 1
# decision = make_decision(current_step, current_budget, cost_per_communication, current_window_cost, x_variance, y_variance, variance_importance_factor)
# print("Decision at step", current_step, ":", decision)
