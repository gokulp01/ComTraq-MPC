import numpy as np
import torch
import matplotlib.pyplot as plt
import model  

env = model.UUV()
waypoints = [5.0, 6.0, 7.0]
env.initialize_particles()
episode_rewards = []

class Node:
    def __init__(self, state, belief, parent=None):
        self.state = state
        self.belief = belief
        self.parent = parent
        self.children = []
        self.visits = 1
        self.value = 0

    def is_fully_expanded(self):
        return len(self.children) == env.num_actions()

    def best_child(self):
        weights = [(child.value / (child.visits + 1e-7)) +
                   np.sqrt(2 * np.log(self.visits) / (child.visits + 1e-7)) for child in self.children]
        return self.children[np.argmax(weights)]
    
    def is_terminal(self):
        if len(self.children==0):
            return True
        else:
            return False

def MCTS(root, simulations):
    for _ in range(simulations):
        node = select_node(root)
        reward = rollout(node)
        backpropagate(node, reward)
    return root.best_child().action


def select_node(node):
    while not node.is_terminal:
        if not node.is_fully_expanded():
            return expand(node)
        else:
            node = node.best_child()
    return node


def expand(node):
    available_actions = set(range(4)) - {child.action for child in node.children}
    action = np.random.choice(list(available_actions))
    next_state, next_belief, reward, done = env.step(action, node.state, waypoints)
    child_node = Node(state=next_state, belief=next_belief, parent=node)
    node.children.append(child_node)
    return child_node


def rollout(node):
    current_state = node.state
    current_belief = node.belief
    total_reward = 0
    done = False
    while not done:
        action = np.random.choice(4)
        next_state, next_belief, reward, done = env.step(action, current_state, waypoints)
        total_reward += reward
        current_belief = next_belief
        current_state = next_state
    return total_reward


def backpropagate(node, reward):
    while node is not None:
        node.visits += 1
        node.value += reward
        node = node.parent




# Main Loop
for episode in range(500):
    state = env.reset()
    belief = env.particles.copy()
    total_reward = 0
    done = False
    while not done:
        root = Node(state=state, belief=belief)
        best_action = MCTS(root, simulations=100)
        next_state, next_belief, reward, done = env.step(best_action, state, waypoints)
        total_reward += reward
        state = next_state
        belief = next_belief
    episode_rewards.append(total_reward)

# Plot results
plt.plot(episode_rewards)
plt.title('Episode Rewards')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.savefig('training_results_mcts.png')
plt.show()
