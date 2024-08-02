import random
import math
import copy
import numpy as np
from control import Car_Dynamics, MPC_Controller, ParticleFilter
initial_budget=10
class State:
    def __init__(self, budget, belief, psi, velocity, final_path, path_index, waypoint, del_var_particles):
        self.budget = budget
        self.belief = belief
        self.comm_cost=1
        self.del_var_particles=del_var_particles
        self.psi = psi  
        self.velocity = velocity
        self.final_path = final_path
        self.path_index = path_index
        self.my_car = Car_Dynamics(self.belief[0], self.belief[1], self.velocity, self.psi, length=4, dt=0.2, pf=ParticleFilter(num_particles=100, init_state=np.array([self.belief[0], self.belief[1], self.psi])))    
        self.MPC_HORIZON = 1
        self.controller = MPC_Controller()
        self.waypoint=waypoint
        self.max_depth = 100
        self.cost=1
        # print(len(self.final_path)) 



    def next_state(self, action):
        # print(f"bud is {self.budget}")

        if action == "communicate" and self.budget > 0:
            next_state = State(self.budget - self.comm_cost, self.belief, self.psi, self.velocity, self.final_path, self.path_index, self.waypoint, self.del_var_particles)
        else:
            next_state= State(self.budget, self.belief, self.psi, self.velocity, self.final_path, self.path_index,self.waypoint, self.del_var_particles)
    
        return next_state

    def update_belief(self, action):
        acc, delta = self.controller.optimize(self.my_car, self.final_path[self.path_index:self.path_index+self.MPC_HORIZON])
        self.my_car.update_state(self.my_car.move(acc,  delta), self.my_car.x, self.my_car.y, self.my_car.psi, action)
        self.belief = [self.my_car.x, self.my_car.y]
        self.psi = self.my_car.psi
        self.velocity = self.my_car.v   
        # print(f"belief mcts: {self.belief}")


    def is_terminal(self, depth):
        # Assuming the game ends when the budget is 0 or waypoint is reached
        return self.waypoint_r() or depth >= min(self.max_depth, len(self.final_path)-self.path_index)  

    def get_possible_actions(self):
        if self.budget > 0:
            return ["communicate", "not_communicate"]
        return ["not_communicate"]

    def waypoint_r(self):
        if math.sqrt((self.belief[0] - self.waypoint[0])**2 + (self.belief[1] - self.waypoint[1])**2) < 1:
            # print("reached")
            return True


    def calculate_reward(self, action):
        # print(f"budget is {self.budget}")
        # uncertainty = np.linalg.norm(self.var_particles)
        budget_utilization = (initial_budget - self.budget) 
        print(f"budget utilization: {budget_utilization}")
        reward = 0
        # if action == "communicate":
        #     reward -= budget_utilization*10

        reward-=math.sqrt((self.belief[0] - self.waypoint[0])**2 + (self.belief[1] - self.waypoint[1])**2)/(math.sqrt((80-41)**2+(90-8)**2))
        # print(f"reward mcts: {reward}")
        if action == "not_communicate":
            
            reward-=(self.del_var_particles[0]+self.del_var_particles[1])*100
        # # print(f"reward mcts2: {reward}") 

        # print(self.del_var_particles)
        # print(f"reward mcts: {reward}")
        if budget_utilization>initial_budget:
            reward-=1000

        if self.waypoint_r():
            print("here")
            reward+=1000
        # print(f"action mcts: {action}")
        # print(f"reward mcts: {reward}")
        # print("---------------")
        return reward

        # return np.linalg.norm(self.del_var_particles)


    def simulate(self, action):
        # print(f"action mcts: {action}")
        return self.calculate_reward(action)
    
    # def clone(self):
    #     return State(self.budget, self.belief.copy(), self.velocity, self.final_path, self.path_index)

        

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
        print(state.budget)
        # Selection
        while node.untried_actions == [] and node.children != []:
            node = node.select_child()
            state = state.next_state(node.action)
            state.update_belief(node.action) 


        # Expansion
        if node.untried_actions != []:

            action = random.choice(node.untried_actions)
            state = state.next_state(action)
            
            state.update_belief(action)
            

            node = node.add_child(action, state)

        # Rollout
        depth = 0
        terminated=False
        while not terminated:
            action = random.choice(state.get_possible_actions())
            # print(f"action mcts: {action}")
            state = state.next_state(action)
            state.update_belief(action)
            state.path_index += 1   
            print(f"belief mcts: {state.belief}")
            depth += 1
            # print(f"depth: {depth}")
            terminated = state.is_terminal(depth)

        # Backpropagation
        result = state.simulate(action)
        while node is not None:
            node.update(result)
            node = node.parent

    return max(root.children, key=lambda c: c.visits).action


# waypoint = [9, 80]  # Target waypoint
# Example usage
# initial_budget = 15  # Example budget
# self, budget, is_true_state, waypoint_reached, belief, del_var_particles, velocity, final_path, path_index
# initial_state = State(initial_budget, False, False, belief=[-0.1, 89.323243243242], )
# best_action = mcts(initial_state, iterations=1000)
# print("Best action:", best_action)
