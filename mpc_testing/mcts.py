import random
import math
waypoint = [90, 80]  # Target waypoint
# Example usage
budget = 6 
class State:
    def __init__(self, is_true_state, waypoint_reached, belief):
        global budget

        self.is_true_state = is_true_state
        self.waypoint_reached = waypoint_reached
        self.belief = belief
        self.comm_cost=1


    def next_state(self, action):
        global budget

        if action == "communicate" and budget > 0:
            print(budget)
            budget -= self.comm_cost
            return State( True, self.waypoint_r(), self.belief)
        return State(False, self.waypoint_r(), self.belief)

    def is_terminal(self):
        global budget

        # Assuming the game ends when the budget is 0 or waypoint is reached
        return budget == 0 or self.waypoint_reached

    def get_possible_actions(self):
        global budget

        if budget > 0:
            return ["communicate", "not_communicate"]
        return ["not_communicate"]

    def waypoint_r(self):
        global budget

        if (self.belief[0] - waypoint[0])**2 + (self.belief[1] - waypoint[1])**2 < 1:
            return True


    def calculate_reward(self, action):
        global budget

        reward = 0
        
        # Reward or penalty based on outcomes, not directly on state accuracy
        if self.waypoint_r():
            reward += 10  # Reward for reaching the waypoint
        else:
            reward -= 0.5  # Penalty for not reaching the waypoint

        # Small reward for saving budget by not communicating
        if action=="not_communicate":
            reward += 0.1

        return reward

    def simulate(self, action):
        global budget

        # This is a dummy simulation, replace with your logic
        return self.calculate_reward(action)
        

class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_actions = state.get_possible_actions()

    def ucb_score(self, total_visits, exploration_const=1.41):
        if self.visits == 0:
            return float('inf')
        return self.wins / self.visits + exploration_const * math.sqrt(math.log(total_visits) / self.visits)

    def select_child(self):
        return max(self.children, key=lambda child: child.ucb_score(self.visits))

    def add_child(self, action, state):
        child_node = Node(state, parent=self, action=action)
        self.children.append(child_node)
        self.untried_actions.remove(action)
        return child_node

    def update(self, result):
        self.visits += 1
        self.wins += result

def mcts(root_state, iterations):
    root = Node(root_state)

    for _ in range(iterations):
        node = root
        state = root_state

        # Selection
        while node.untried_actions == [] and node.children != []:
            node = node.select_child()
            state = state.next_state(node.action)

        # Expansion
        if node.untried_actions != []:
            action = random.choice(node.untried_actions)
            state = state.next_state(action)
            node = node.add_child(action, state)

        # Rollout
        while not state.is_terminal():
            action = random.choice(state.get_possible_actions())
            state = state.next_state(action)

        # Backpropagation
        result = state.simulate(action)
        while node is not None:
            node.update(result)
            node = node.parent

    return max(root.children, key=lambda c: c.visits).action


 # Example budget
# initial_budget = 6  # Example budget
initial_state = State( False, False, belief=[-0.1, 89.323243243242])
best_action = mcts(initial_state, iterations=1000)
print("Best action:", best_action)
