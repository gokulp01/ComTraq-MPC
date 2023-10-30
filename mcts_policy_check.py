import pickle
import model
import random
# Load the policy dictionary from the saved file
with open('mcts_policy_3.pkl', 'rb') as f:
    policy_dict = pickle.load(f)

# Function to select an action based on the policy
def select_action(state):
    # Convert the state to the same format as the keys in the policy_dict
    state_key = tuple(state)
    
    if state_key in policy_dict:
        # Retrieve the visit counts of actions for the given state
        action_visit_counts = policy_dict[state_key]
        
        # Choose the action with the highest visit count
        best_action = max(action_visit_counts, key=action_visit_counts.get)
        return best_action
    else:
        default_action=random.randint(0,4)
        print("defaulted")
        # Handle the case when the state is not in the policy dictionary
        # You can choose a default action or use some other strategy
        return default_action

# Example usage:
# Given a state `current_state`, select an action using the policy

env=model.UUV()
s=env.reset()
waypoints=[5,6,7]
done=False
while not done:
    selected_action = select_action(s)
    next_state, next_belief, reward, done=env.step(selected_action, s, waypoints)
    print(f"State {s}, Reward {reward}, Action {selected_action}, Done {done}, N State {next_state}")
    print("-"*30)
    s=next_state


# Now, `selected_action` contains the action chosen based on the saved policy weights
print(f"Selected action: {selected_action}")
