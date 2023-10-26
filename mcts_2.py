import numpy as np
import torch
import matplotlib.pyplot as plt
import model  

env = model.UUV()
waypoints = [5.0, 6.0, 7.0]
env.initialize_particles()
episode_rewards = []


action_dim=env.num_actions
class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.total_reward = 0

    def is_fully_expanded(self):
        return len(self.children) == action_dim

    def best_child(self, c_param=1.0):
        choices_weights = [
            (child.total_reward / (child.visit_count + 1e-7)) +
            c_param * np.sqrt((2 * np.log(self.visit_count + 1e-7) / (child.visit_count + 1e-7)))
            for child in self.children.values()
        ]
        return list(self.children.values())[np.argmax(choices_weights)]

    def expand(self, action, next_state):
        child = MCTSNode(next_state, parent=self)
        self.children[action] = child
        return child


def mcts_rollout(node):
    current_node = node
    for _ in range(10):
        if current_node.is_fully_expanded():
            current_node = current_node.best_child()
        else:
            # Expand
            available_actions = set(range(action_dim)) - set(current_node.children.keys())
            action = np.random.choice(list(available_actions))
            next_state, reward, done = env.step_rollout(action, current_node.state, waypoints)
            # next_belief_state=env.most_frequent_state()
            # if done:
            #     return reward
            current_node = current_node.expand(action, next_state)
            return reward


def mcts_search(root, iterations):
    for _ in range(iterations):
        # Selection & Expansion
        node = root
        while node.is_fully_expanded():
            # print("tre")
            node = node.best_child()

        # Simulation
        reward = mcts_rollout(node)

        # Backpropagation
        while node is not None:
            node.visit_count += 1
            node.total_reward += reward
            node = node.parent

    # Return the action of the best child of the root node
    return max(root.children.keys(), key=lambda action: root.children[action].visit_count)


# Pure MCTS loop
pure_mcts_episode_rewards = []
pure_mcts_episode_lengths = []
max_episodes=100
for i_episode in range(1, max_episodes + 1):
    print(i_episode)
    total_reward = 0
    state = env.reset()
    belief=env.particles.copy()
    belief_state=env.most_frequent_state()
    belief_state[4] = state[4]
    done = False
    steps = 0
    while not done:
        steps += 1
        print(f"State{state}, B state {belief_state}")
        root = MCTSNode(belief_state)
        action = mcts_search(root, iterations=10)
        next_state, next_belief, reward, done = env.step(action, state, waypoints)
        next_belief_state=env.most_frequent_state()
        next_belief_state[4] = next_state[4]
        total_reward += reward
        state = next_state
        belief_state=next_belief_state
        print(f"N state{next_state}, Reward {reward}, Done {done}")
    pure_mcts_episode_rewards.append(total_reward)
    pure_mcts_episode_lengths.append(steps)
    print(f"Steps{steps}")
    # Logging
    if i_episode % 5 == 0:
        print('Pure MCTS Episode {} \t Avg length: {} \t Avg reward: {}'.format(
            i_episode, np.mean(pure_mcts_episode_lengths[-10:]), np.mean(pure_mcts_episode_rewards[-10:]))
        )

# Plot results for Pure MCTS
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(pure_mcts_episode_rewards)
plt.title('Pure MCTS Episode Rewards')
plt.xlabel('Episode')
plt.ylabel('Total Reward')

plt.subplot(1, 2, 2)
plt.plot(pure_mcts_episode_lengths)
plt.title('Pure MCTS Episode Lengths')
plt.xlabel('Episode')
plt.ylabel('Length')

plt.tight_layout()
plt.savefig('pure_mcts_training_results.png')
plt.show()
