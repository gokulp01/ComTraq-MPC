# %%
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import model  # Make sure to import your custom environment
import os

from model_ppo import TrainModel_PPO
from utils_ppo import import_train_configuration, set_train_path



config = import_train_configuration(config_file='training_settings_ppo.ini')
def compress_state(state):
    # Assuming `state` is a 2D array of shape (100000, 5)
    compressed_state = np.mean(state, axis=0)
    return compressed_state

# Define the Actor-Critic network

class TrainModel_PPO(nn.Module):
    def __init__(self, learning_rate_actor, leraning_rate_critic, input_dim, output_dim, eps_clip):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr_actor = learning_rate_actor
        self.lr_critic = leraning_rate_critic
        self.eps_clip = eps_clip
        self.actor = self._build_actor_model()
        self.critic = self._build_critic_model()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam([
            {'params': self.actor.parameters(), 'lr': self.lr_actor},
            {'params': self.critic.parameters(), 'lr': self.lr_critic}
            ])

    def _build_actor_model(self):
        layers = []
        layers.append(nn.Linear(self.input_dim, 256))
        layers.append(nn.Tanh())
        layers.append(nn.Linear(256, 256))
        layers.append(nn.Tanh())
        layers.append(nn.Linear(256, self.output_dim))
        layers.append(nn.Softmax(dim=-1))
        return nn.Sequential(*layers)
    
    def _build_critic_model(self):
        layers = []
        layers.append(nn.Linear(self.input_dim, 256))
        layers.append(nn.Tanh())
        layers.append(nn.Linear(256, 256))
        layers.append(nn.Tanh())
        layers.append(nn.Linear(256, 1))
        return nn.Sequential(*layers)
    
    def forward(self):
        raise NotImplementedError
    
    def act(self,state):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy
    
    def train(self, old_states, old_actions, old_logprobs, rewards, advantages):
        action_logprobs, state_values, dist_entropy = self.evaluate(old_states,old_actions)

        state_values = torch.squeeze(state_values)
        ratios = torch.exp(action_logprobs - old_logprobs.detach())

        surrogate_1 = ratios*advantages
        surrogate_2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip)*advantages

        loss = -torch.min(surrogate_1, surrogate_2) + 0.5 * self.criterion(state_values, rewards) - 0.01 * dist_entropy

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()
    
    def save_actor_model(self, path):
        torch.save(self.actor.state_dict(), os.path.join(path, 'trained_actor_model.pt'))

    def load_actor_model(self, path):
        self.actor.load_state_dict(torch.load(path))
        self.actor.eval()

# Hyperparameters
# state_dim = 5  # Should be the processed state dimension, not the raw particle count
# action_dim = 4  # Number of actions
# hidden_dim = 256  # Number of hidden units
# lr = 0.002
# betas = (0.9, 0.999)
# gamma = 0.99
# K_epochs = 10
# eps_clip = 0.2
# max_episodes = 500
# max_timesteps = 300
# episode_rewards = []
# episode_lengths = []
waypoints=[5.0,6.0,7.0]
#
# %%

# %%

env = model.UUV()
env.initialize_particles()

# Load the policy
state_dim = 5
action_dim = 4
hidden_dim = 256
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# policy = TrainModel_PPO(state_dim, action_dim, hidden_dim).to(device)

policy = TrainModel_PPO(
config['learning_rate_actor'], 
config['learning_rate_critic'],
input_dim=config['num_states'], 
output_dim=config['num_actions'],
eps_clip=config['eps_clip']
)

# Load the saved weights
policy.load_state_dict(torch.load('model_weights/policy_ep_280.pt', map_location=device))

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

        print(f"reward is {reward}")
        print(f"state is {state}")
        print(f"n state is {next_state}")
        print(f"belief is {belief_c}")
        print(f"n belief is {compress_state(next_belief)}")
        print("=============================================")
        belief = next_belief
        state = next_state

    test_rewards.append(episode_reward)
    print(f'Episode {i_episode} reward: {episode_reward}')

avg_reward = np.mean(test_rewards)
print(f'Average reward over {num_test_episodes} episodes: {avg_reward}')


# %%



