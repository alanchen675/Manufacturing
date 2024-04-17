import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np


class PPOAgent:
    """
    Agent model for the problem
    """
    def __init__(self, n_observations, n_actions, chkpt_dir, hidden_dims=128):
        self.actor = Actor(n_observations, n_actions, hidden_dims)
        self.critic = Critic(n_observations, n_actions, hidden_dims)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.lr)
        self.cov_var = torch.full(size=(n_actions,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)
        self.chkpt_dir = chkpt_dir

    def save_model(self, filename):
        if not os.path.exists(self.chkpt_dir):
            os.mkdir(self.chkpt_dir)
        torch.save(self.actor.state_dict(), f'{self.chkpt_dir}/{filename}')
        torch.save(self.critic.state_dict(), f'{self.chkpt_dir}/{filename}')

    def load_model(self, filename):
        self.actor.load_state_dict(torch.load(f'{self.chkpt_dir}/{filename}'))
        self.critic.load_state_dict(torch.load(f'{self.chkpt_dir}/{filename}'))

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

    def evaluate(self, batch_obs, batch_acts):
        V = self.critic(batch_obs).squeeze()
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)
        return V, log_probs

    def learn(self, batch_obs, batch_acts, batch_log_probs, batch_rtgs, n_itr):
        """
        Learning for an agent
        """
        V, _ = self.evaluate(batch_obs, batch_acts)
        A_k = batch_rtgs - V.detach()
        A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

        for _ in range(n_itr):
            V, curr_log_probs = self.evaluate(batch_obs, batch_acts)
            ratios = torch.exp(curr_log_probs - batch_log_probs)
            surr1 = ratios * A_k
            surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k
            actor_loss = (-torch.min(surr1, surr2)).mean()
            critic_loss = nn.MSELoss()(V, batch_rtgs)
            self.actor_optim.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_optim.step()
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()
            #self.logger['actor_losses'][f'agent{ag+1}'].append(actor_loss.detach())
        #self._log_summary(ag)

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



