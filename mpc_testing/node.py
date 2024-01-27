import random
from math import log, sqrt
import random
import copy
import numpy as np

initial_budget = 30  # Example budget
waypoint = [90, 80]  # Target waypoint



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



class Node:

	def __init__(self, state):
		self.state = state
		self.win_value = 0
		self.policy_value = None
		self.visits = 0
		self.parent = None
		self.children = []
		self.expanded = False
		self.player_number = None
		self.discovery_factor = 0.35

	def update_win_value(self, value):
		self.win_value += value
		self.visits += 1

		if self.parent:
			self.parent.update_win_value(value)

	def update_policy_value(self, value):
		self.policy_value = value

	def add_child(self, child):
		self.children.append(child)
		child.parent = self

	def add_children(self, children):
		for child in children:
			self.add_child(child)

	def get_preferred_child(self, root_node):
		best_children = []
		best_score = float('-inf')

		for child in self.children:
			score = child.get_score(root_node)

			if score > best_score:
				best_score = score
				best_children = [child]
			elif score == best_score:
				best_children.append(child)

		return random.choice(best_children)

	def get_score(self, root_node):
		discovery_operand = self.discovery_factor * (self.policy_value or 1) * sqrt(log(self.parent.visits) / (self.visits or 1))

		win_multiplier = 1 if self.parent.player_number == root_node.player_number else -1
		win_operand = win_multiplier * self.win_value / (self.visits or 1)

		self.score = win_operand + discovery_operand

		return self.score

	def is_scorable(self):
		return self.visits or self.policy_value != None