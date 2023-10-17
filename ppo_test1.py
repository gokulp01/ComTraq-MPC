import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import model  # Make sure to import your custom environment
import os

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

# Create a directory for saving model weights
os.makedirs('model_weights', exist_ok=True)
# Set up environment
env = model.UUV()  # Initialize your environment
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set up policy network
policy = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
optimizer = optim.Adam(policy.parameters(), lr=lr, betas=betas)
waypoints=[5.0,6.0,7.0]
env.initialize_particles()


torch.autograd.set_detect_anomaly(True)


# Training loop
for i_episode in range(1, max_episodes+1):
    print(i_episode)
    total_reward = 0 
    state = env.reset()
    belief=env.particles.copy()
    done=False
    for t in range(max_timesteps):
        # Select action
        belief_c=compress_state(belief)
        action_probs, _ = policy(belief_c)
        dist = Categorical(action_probs)
        action = dist.sample()

        # Take action in environment
        next_state, next_belief, reward, done = env.step(action.cpu().numpy(), state, waypoints)
        total_reward += reward
        next_belief_c=compress_state(next_belief)
        # Calculate advantage
        _, value = policy(belief_c)
        _, next_value = policy(next_belief_c)
        value = value.detach()
        next_value = next_value.detach()
        advantage = reward + gamma * next_value * (1-int(done)) - value
        advantage = advantage.detach()  # Don't propagate gradients through advantage

        # Update policy
# Update policy
        for k in range(K_epochs):
            current_action_probs, current_value = policy(belief_c)
            current_dist = Categorical(current_action_probs)

            # PPO's policy loss
            ratio = torch.exp(current_dist.log_prob(action) - dist.log_prob(action))
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = 0.5 * (reward + gamma * next_value * (1-int(done)) - current_value).pow(2).mean()

            # Total loss
            loss = policy_loss + value_loss

            # Backpropagation
            optimizer.zero_grad()
            # Retain graph for all but the last iteration
            retain_graph_flag = k < K_epochs - 1
            loss.backward(retain_graph=retain_graph_flag)
        optimizer.step()
        # print(f"next belief is {next_belief_c}")


        if done:
            break

        belief = next_belief
        state=next_state
    episode_rewards.append(total_reward)  # Save the total episode reward
    episode_lengths.append(t) 

    # Logging
    if i_episode % 10 == 0:
        print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, np.mean(episode_lengths[-10:]), np.mean(episode_rewards[-10:])))
        # Save model weights
        torch.save(policy.state_dict(), f'model_weights/policy_ep_{i_episode}.pt')

# Close environment
# env.close()

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(episode_rewards)
plt.title('Episode Rewards')
plt.xlabel('Episode')
plt.ylabel('Total Reward')

plt.subplot(1, 2, 2)
plt.plot(episode_lengths)
plt.title('Episode Lengths')
plt.xlabel('Episode')
plt.ylabel('Length')

plt.tight_layout()
plt.savefig('training_results.png')  # Save the figure as a file
plt.show()
