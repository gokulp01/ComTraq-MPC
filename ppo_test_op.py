# %%
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import model  # Make sure to import your custom environment
import os

# %%
def compress_state(state):
    # Assuming `state` is a 2D array of shape (100000, 5)
    compressed_state = np.mean(state, axis=0)
    return compressed_state

# Define the Actor-Critic network
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        state = torch.from_numpy(state).float()
        action_probs = self.actor(state)
        value = self.critic(state)
        return action_probs, value

# Hyperparameters
state_dim = 5  # Should be the processed state dimension, not the raw particle count
action_dim = 74  # Number of actions
hidden_dim = 256  # Number of hidden units
lr = 0.002
betas = (0.9, 0.999)
gamma = 0.99
K_epochs = 10
eps_clip = 0.2
max_episodes = 500
max_timesteps = 300
episode_rewards = []
episode_lengths = []
waypoints=[5.0,6.0,7.0]

# %%

# %%

env = model.UUV()
env.initialize_particles()

# Load the policy
state_dim = 5
action_dim = 74
hidden_dim = 256
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
policy = ActorCritic(state_dim, action_dim, hidden_dim).to(device)

# Load the saved weights
policy.load_state_dict(torch.load('model_weights/policy_ep_190.pt', map_location=device))

# Evaluate the policy
num_test_episodes = 1  # for instance, evaluate for 100 episodes
test_rewards = []

for i_episode in range(1, num_test_episodes+1):
    state = env.reset()
    belief = env.particles.copy()
    episode_reward = 0
    done = False

    while not done:
        belief_c = compress_state(belief)
        with torch.no_grad():  # Don't compute gradient for evaluations
            action_probs, _ = policy(belief_c)
        dist = Categorical(action_probs)
        action = dist.sample()

        print(f"action taken is {action.item()}")

        # env.render()  # Remove this line if your environment doesn't support rendering

        next_state, next_belief, reward, done = env.step(action.cpu().numpy(), state, waypoints)
        episode_reward += reward

        print(f"state is {state}")
        print(f"n state is {next_state}")
        print(f"belief is {belief_c}")
        print(f"n belief is {compress_state(next_belief)}")

        belief = next_belief
        state = next_state

    test_rewards.append(episode_reward)
    print(f'Episode {i_episode} reward: {episode_reward}')

avg_reward = np.mean(test_rewards)
print(f'Average reward over {num_test_episodes} episodes: {avg_reward}')


# %%



