import pulp  import numpy as np

def optimize_communication(current_state, true_state, waypoint, comm_cost, budget_remaining):
    """
    Optimize the decision to communicate or not, considering the waypoint.

    :param current_state: The current belief state (x, y, theta).
    :param true_state: The true state (x, y, theta).
    :param waypoint: The target waypoint (x, y).
    :param comm_cost: The constant cost of communication.
    :param budget_remaining: The remaining budget for communication.
    :return: A decision to communicate (1) or not (0).
    """
    problem = pulp.LpProblem("Communication_Optimization", pulp.LpMinimize)

    communicate = pulp.LpVariable('Communicate', 0, 1, cat='Binary')

    state_difference = sum((current_state[i] - true_state[i])**2 for i in range(3))
    waypoint_difference = sum((current_state[i] - waypoint[i])**2 for i in range(2))  # x and y only
    print(waypoint_difference)
    penalty = 100  # Penalty for not communicating
    objective = state_difference * communicate + waypoint_difference *1000 * (1 - communicate) + penalty * (1 - communicate)
    problem += objective

    problem += communicate * comm_cost <= budget_remaining

    problem.solve(pulp.PULP_CBC_CMD(msg=False))

    decision = pulp.value(communicate)
    return decision

budget = 100
budget_remaining = budget
comm_cost = 15

def get_current_belief_state():
    return [np.random.uniform(8, 12), np.random.uniform(8, 12), np.random.uniform(8, 12)]

def get_true_state():
    return [10, 10, 10]

waypoint = [15, 15]
steps = 15
decision_list = []

while True:
    current_state = get_current_belief_state()
    true_state = get_true_state()

    decision = optimize_communication(current_state, true_state, waypoint, comm_cost, budget_remaining)

    decision_list.append(decision)
    if decision == 1:
        budget_remaining -= comm_cost

    steps -= 1
    print("Decision:", decision)
    print("Steps left:", steps)
    print("Budget remaining:", budget_remaining)
    print("------------------")

    if steps <= 0 or budget_remaining <= 0:
        break

np.save('decision_list.npy', np.array(decision_list))
