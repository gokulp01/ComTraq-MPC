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
    while True:
        if current_node.is_fully_expanded():
            current_node = current_node.best_child()
        else:
            # Expand
            available_actions = set(range(action_dim)) - set(current_node.children.keys())
            action = np.random.choice(list(available_actions))
            next_state, next_belief, reward, done = env.step_rollout(action, current_node.state, waypoints)
            next_belief_state=env.rollout_most_frequent_state()
            if done:
                return reward
            current_node = current_node.expand(action, next_belief_state)
            # return reward


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


def prune_tree(visit_threshold=7):
    global state_to_node_map
    state_to_node_map = {state: node for state, node in state_to_node_map.items() if node.visit_count >= visit_threshold}


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) 

def softmax_action_selection(belief_state):
    """Select an action based on a softmax of visit counts."""
    visit_counts = np.array(list(policy_dict[tuple(belief_state)].values()))
    probabilities = softmax(visit_counts)
    actions = list(policy_dict[tuple(belief_state)].keys())
    return np.random.choice(actions, p=probabilities)

state_to_node_map = {}
pure_mcts_episode_rewards = []
pure_mcts_episode_lengths = []
max_episodes = 1

pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

policy_dict = {}
pure_mcts_episode_rewards = []
pure_mcts_episode_lengths = []

# Start training loop from the next episode
start_episode = len(pure_mcts_episode_rewards) + 1
pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

for i_episode in range(start_episode, max_episodes + 1):
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
        num_processes = 8
        env.rollout_particles = env.particles.copy()
        results = pool.map(parallel_mcts_search, [(root, 30 // num_processes) for _ in range(num_processes)])  # Double the simulations

        for res in results:
            for state_key, action_visit_counts in res:
                if state_key in policy_dict:
                    for action, count in action_visit_counts.items():
                        policy_dict[state_key][action] = policy_dict[state_key].get(action, 0) + count
                else:
                    policy_dict[state_key] = action_visit_counts

        action = softmax_action_selection(belief_state)
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

    # 4. Use pruning at regular intervals
    if i_episode % 1 == 0:
        prune_tree()  # Prune the tree every 10 episodes to remove less visited nodes

        # Save state
        with open('mcts_saved_state_new_eb2.pkl', 'wb') as f:
            pickle.dump({
                'policy_dict': policy_dict,
                'episode_rewards': pure_mcts_episode_rewards,
                'episode_lengths': pure_mcts_episode_lengths
            }, f)

pool.close()
pool.join()

with open('mcts_policy_5.pkl', 'wb') as f:
    pickle.dump(policy_dict, f)
