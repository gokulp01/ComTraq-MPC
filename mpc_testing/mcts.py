import random
import math
import copy
import numpy as np

initial_budget = 30  # Example budget

class State:
    def __init__(self, budget, is_true_state, waypoint_reached, belief, del_var_particles):
        self.budget = budget
        self.is_true_state = is_true_state
        self.waypoint_reached = waypoint_reached
        self.belief = belief
        self.comm_cost=1
        self.del_var_particles=del_var_particles

    def next_state(self, action):
        if action == "communicate" and self.budget > 0:
            # print(self.budget)
            return State(self.budget - self.comm_cost, True, self.waypoint_r(), self.belief, self.del_var_particles)
        
        return State(self.budget, False, self.waypoint_r(), self.belief, self.del_var_particles)

    def is_terminal(self):
        # Assuming the game ends when the budget is 0 or waypoint is reached
        return self.budget == 0 or self.waypoint_reached

    def get_possible_actions(self):
        if self.budget > 0:
            return ["communicate", "not_communicate"]
        return ["not_communicate"]

    def waypoint_r(self):
        if (self.belief[0] - waypoint[0])**2 + (self.belief[1] - waypoint[1])**2 < 1:
            return True


    def calculate_reward(self, action):
        # uncertainty = np.linalg.norm(self.var_particles)
        budget_utilization = (initial_budget - self.budget) / initial_budget

        reward = 0
        reward-=(self.del_var_particles[0]+self.del_var_particles[1])*100
        if action == "communicate":
            # print(budget_utilization)
            reward -= budget_utilization*100
        
        # if self.waypoint_r():
        #     reward += 10  # Reward for reaching the waypoint
        # else:
        #     reward -= 0.2  # Penalty for not reaching the waypoint

# rewrard based on change in uncertainty
        

        # if action == "not_communicate":
        #     reward -= uncertainty *10
        # elif action == "communicate":
        #     reward -= budget_utilization 

        return reward
        # return np.linalg.norm(self.del_var_particles)


    def simulate(self, action):
        return self.calculate_reward(action)
    
    def clone(self):
        return State(self.budget, self.is_true_state, self.waypoint_reached, self.belief.copy(), self.del_var_particles)

        

class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_actions = state.get_possible_actions()

    def ucb_score(self, total_visits, exploration_const=0.2):
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

    for i in range(iterations):
        # print(f"Iteration {i+1}: Starting with budget {root_state.budget}")
        node = root
        state = copy.deepcopy(root_state)  # Deep copy the root state

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


waypoint = [90, 80]  # Target waypoint
# Example usage
# initial_budget = 6  # Example budget
# initial_state = State(initial_budget, False, False, belief=[-0.1, 89.323243243242])
# best_action = mcts(initial_state, iterations=1000)
# print("Best action:", best_action)
