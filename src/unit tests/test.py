def online_decision_making(total_steps, initial_budget, cost_per_communication):
    # Initialize variables
    budget = initial_budget
    communicate_sequence = [0] * total_steps  # 0: no_communicate, 1: communicate
    knowledge_state = [1] + [0] * (total_steps - 1)  # Assume perfect initial knowledge

    for t in range(1, total_steps):
        # Check if we have budget and if the previous action was not communication
        if budget >= cost_per_communication and communicate_sequence[t - 1] == 0:
            # Decision to communicate based on specific criteria
            # For simplicity, this example uses a placeholder condition for communication
            # You should replace it with your system-specific condition
            # E.g., high deviation from expected state, low current knowledge state, critical step, etc.

            # Placeholder for a decision-making condition
            if (
                knowledge_state[t - 1] == 0
            ):  # Example condition: previous knowledge state is low
                communicate_sequence[t] = 1  # Decide to communicate
                budget -= cost_per_communication  # Update budget
                knowledge_state[
                    t
                ] = 1  # Update knowledge state to perfect after communication
            else:
                # No need to communicate, carry forward the knowledge state
                knowledge_state[t] = knowledge_state[t - 1]
        else:
            # Either budget is insufficient, or we just communicated
            knowledge_state[t] = knowledge_state[
                t - 1
            ]  # Carry forward the knowledge state

    return communicate_sequence


# Example usage
total_steps = 10
initial_budget = 5  # Assuming each communication costs 1
cost_per_communication = 1
communicate_sequence = online_decision_making(
    total_steps, initial_budget, cost_per_communication
)

print("Communication Sequence:", communicate_sequence)
