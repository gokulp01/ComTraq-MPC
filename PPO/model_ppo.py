import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import model
import os
import sys
from utils_ppo import import_train_configuration
from rollout_buffer import RolloutBuffer
from torch.distributions import Categorical
from torch.distributions import MultivariateNormal


config = import_train_configuration(config_file='ppo/training_settings_ppo.ini')

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


# class TestModel_PPO:
#     def __init__(self, input_dim, model_path):
#         self.input_dim = input_dim
#         self.model = self._load_my_model(model_path)


#     def _load_my_model(self, model_folder_path):
#         model_file_path = os.path.join(model_folder_path, 'trained_actor_model.pt')
        
#         if os.path.isfile(model_file_path):
#             loaded_model_state_dict = torch.load(model_file_path)
#             model = TrainModel_PPO(input_dim=self.input_dim, output_dim=config['num_actions'], learning_rate_actor=config['learning_rate_actor'], leraning_rate_critic=config['learning_rate_critic'], eps_clip=config['eps_clip']).actor
#             model.load_state_dict(loaded_model_state_dict)
#             model.eval()
#             return model
#         else:
#             sys.exit("Model number not found")

#     def predict_one(self, state):
#         state = np.reshape(state, [1, self.input_dim])
#         state = torch.tensor(state, dtype=torch.float32)
#         with torch.no_grad():
#             return self.model(state).numpy()
        