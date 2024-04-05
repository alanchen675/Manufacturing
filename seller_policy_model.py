import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np


class PPOAgent:
    def __init__(self, hyper_para, n_observations, n_actions, hidden_dims=128):
        self._init_hyperparameter()
        self.actor = Actor(n_observations, n_actions, hidden_dims)
        self.critic = Critic(n_observations, n_actions, hidden_dims)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.actor_optim = optim.Adam(self.critic.parameters(), lr=self.lr)
        self.cov_var = torch.full(size=(n_actions,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

    def _init_hyperparameters(self):
        self.timesteps_per_batch = 2048            # timesteps per batch
        self.max_timesteps_per_episode = 200      # timesteps per episode
        self.gamma = 0.99 #discount factor
        self.n_updates_per_iteration = 10
        self.lr = 3e-4
        self.clip = 0.2
        self.save_freq = 1
        self.seed = None
        for param, val in self.hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))
            # Sets the seed if specified
        if self.seed != None:
            # Check if our seed is valid first
            assert(type(self.seed) == int)
            # Set the seed
            torch.manual_seed(self.seed)
            print(f"Successfully set seed to {self.seed}")

    def save_model(self):
        if not os.path.exists(self.chkpt_dir):
            os.mkdir(self.chkpt_dir)
        torch.save(self.actor.state_dict(), f'{self.chkpt_dir}/actor.pth')
        torch.save(self.critic.state_dict(), f'{self.chkpt_dir}/critic.pth')

    def load_model(self):
        self.actor.load_state_dict(torch.load(f'{self.chkpt_dir}/actor.pth'))
        self.critic.load_state_dict(torch.load(f'{self.chkpt_dir}/critic.pth'))

    def get_action(self, obs):
        """
        Given the state, output the actions
        """
        mean = self.actor(obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        # Sample an action from the distribution and get its log prob
        action = dist.sample()
        log_prob = dist.log_prob(action)
  
        # Return the sampled action and the log prob of that action
        return action, log_prob

class Critic(nn.Module):
    def __init__(self, n_observations, n_actions, hidden_dims=128):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(n_observations, hidden_dims)
        self.layer2 = nn.Linear(hidden_dims, hidden_dims)
        self.layer3 = nn.Linear(hidden_dims, 1)

    def forward(self, x):

        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)

        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        value = self.layer3(x)

        return value 

class Actor(nn.Module):
    def __init__(self, n_observations, n_actions, hidden_dims=128):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(n_observations, hidden_dims)
        self.layer2 = nn.Linear(hidden_dims, hidden_dims)
        self.layer3 = nn.Linear(hidden_dims, n_actions)

    def forward(self, x):

        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)

        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        output = F.softmax(self.layer3(x))

        return output



