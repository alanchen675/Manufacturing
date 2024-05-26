import torch
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import MultivariateNormal
from config import SELLER, BUYER, TRANSFORM, stages
from utils import get_result_folder

class PPOAgent:
    """
    Agent model for the problem
    """
    def __init__(self, n_observations, n_actions, chkpt_dir, hidden_dims=128, lr=0.01):
        self.actor = Actor(n_observations, n_actions, hidden_dims)
        self.critic = Critic(n_observations, n_actions, hidden_dims)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)
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
        Learn for an agent
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

class AgentPool:
    """
    Agent pool for all the three types in the problem
    """
    def __init__(self, agents, num_commodities=1, history_length=1):
        self.agents = int(agents)
        self.result_folder = get_result_folder()
        self._init_agents(num_commodities, history_length)

    def _init_agents(self, num_agents, num_commodities=1, history_length=1):
        """
        Initialize the seller, buyer, transformation agents 
        """
        self.seller_obs_dim = num_commodities * history_length * (1+num_agents*(12+num_agents*4))
        self.buyer_obs_dim = self.seller_obs_dim + 2*num_commodities*num_agents
        self.transer_obs_dim = num_commodities*(4*num_agents+3)
        self.seller_act_dim = 4*num_commodities
        self.buyer_act_dim = 2*num_commodities*(num_agents-1)+num_commodities
        self.transer_act_dim = 2*num_commodities
        self.seller_cov_var = torch.full(size=(self.seller_act_dim,), fill_value=0.5)
        self.buyer_cov_var = torch.full(size=(self.buyer_act_dim,), fill_value=0.5)
        self.trans_cov_var = torch.full(size=(self.transer_act_dim,), fill_value=0.5)
        self.seller_cov_mat = torch.diag(self.seller_cov_var)
        self.buyer_cov_mat = torch.diag(self.buyer_cov_var)
        self.trans_cov_mat = torch.diag(self.trans_cov_var)
        self.agent_pools = [None for _ in range(len(stages))] 

        chkpt_path = '{}/chpkt/{}_{}'
        self.agent_pools[SELLER] = [PPOAgent(self.seller_obs_dim,\
                self.seller_act_dim, chkpt_path.format(self.result_folder, 'seller', ag)) for ag in range(self.agents)]
        self.agent_pools[BUYER] = [PPOAgent(self.buyer_obs_dim,\
                self.buyer_act_dim, chkpt_path.format(self.result_folder, 'buyer', ag)) for ag in range(self.agents)]
        self.agent_pools[TRANSFORM] = [PPOAgent(self.transer_obs_dim,\
                self.transer_act_dim, chkpt_path.format(self.result_folder, 'trans', ag)) for ag in range(self.agents)]

    def get_actions(self, obs, agent_type):
        """
        Given the state, output all kinds of quantities 
        """
        actions = []
        log_probs = []
        for i in range(self.agents):
            act, logp = self.agent_pools[agent_type][i].get_action(obs[i,:])
            actions.append(act)
            log_probs.append(logp)

        return np.array(actions), np.array(log_probs)

    def evaluate(self, batch_obs, batch_acts, agent_type, agent_id):
        return self.agent_pools[agent_type][agent_id].evaluate(batch_obs, batch_acts)

    def learn(self, batch_obs, batch_acts, batch_logprobs, batch_rtgs, agent_type, agent_id, n_itr):
        self.agent_pools[agent_type][agent_id].learn(batch_obs, batch_acts, batch_logprobs, batch_rtgs, n_itr) 

    def save_model(self, agent_type, agent_id, filename):
        self.agent_pools[agent_type][agent_id].save_model(filename)

    def load_model(self, agent_type, agent_id, filename):
        self.agent_pools[agent_type][agent_id].load_model(filename)


