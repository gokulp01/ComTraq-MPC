import numpy as np
import matplotlib.pyplot as plt
import model
import pickle
import multiprocessing

env = model.UUV()
waypoints = [5.0, 6.0, 7.0]
env.initialize_particles()
episode_rewards = []

policy_dict = {}

action_dim = env.num_actions


class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.total_reward = 0

    def is_fully_expanded(self):
        return len(self.children) == action_dim

    def best_child(self, c_param=5.0):
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
    for _ in range(30):
        if current_node.is_fully_expanded():
            current_node = current_node.best_child()
        else:
            # Expand
            available_actions = set(range(action_dim)) - set(current_node.children.keys())
            action = np.random.choice(list(available_actions))
            next_state, reward, done = env.step_rollout(action, current_node.state, waypoints)
            current_node = current_node.expand(action, next_state)
            return reward


def parallel_mcts_search(params):
    root, iterations = params
    results = []
    for _ in range(iterations):
        node = root
        while node.is_fully_expanded():
            node = node.best_child()
        reward = mcts_rollout(node)
        while node is not None:
            node.visit_count += 1
            node.total_reward += reward
            node = node.parent
        state_key = tuple(root.state)
        action_visit_counts = {action: child.visit_count for action, child in root.children.items()}
        results.append((state_key, action_visit_counts))
    return results


def get_or_create_node(state):
    state_key = tuple(state)
    if state_key not in state_to_node_map:
        node = MCTSNode(state)
        state_to_node_map[state_key] = node
    return state_to_node_map[state_key]


def prune_tree(visit_threshold=5):
    global state_to_node_map
    state_to_node_map = {state: node for state, node in state_to_node_map.items() if node.visit_count >= visit_threshold}


state_to_node_map = {}
pure_mcts_episode_rewards = []
pure_mcts_episode_lengths = []
max_episodes = 100

pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

for i_episode in range(1, max_episodes + 1):
    print(i_episode)
    total_reward = 0
    state = env.reset()
    belief = env.particles.copy()
    belief_state = env.most_frequent_state()
    belief_state[4] = state[4]
    done = False
    steps = 0

    while not done:
        steps += 1
        root = get_or_create_node(belief_state)
        num_processes = 4
        results = pool.map(parallel_mcts_search, [(root, 10 // num_processes) for _ in range(num_processes)])

        for res in results:
            for state_key, action_visit_counts in res:
                if state_key in policy_dict:
                    for action, count in action_visit_counts.items():
                        policy_dict[state_key][action] = policy_dict[state_key].get(action, 0) + count
                else:
                    policy_dict[state_key] = action_visit_counts

        action = max(policy_dict[tuple(belief_state)].keys(), key=lambda action: policy_dict[tuple(belief_state)][action])
        next_state, next_belief, reward, done = env.step(action, state, waypoints)
        next_belief_state = env.most_frequent_state()
        next_belief_state[4] = next_state[4]
        total_reward += reward
        state = next_state
        belief_state = next_belief_state

    pure_mcts_episode_rewards.append(total_reward)
    pure_mcts_episode_lengths.append(steps)
    print(f"Steps{steps}")
    print('Pure MCTS Episode {} \t Avg length: {} \t Avg reward: {}'.format(
        i_episode, np.mean(pure_mcts_episode_lengths[-10:]), np.mean(pure_mcts_episode_rewards[-10:]))
    )

pool.close()
pool.join()

with open('mcts_policy_2.pkl', 'wb') as f:
    pickle.dump(policy_dict, f)
