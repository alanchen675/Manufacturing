import torch
import time
import warnings
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import MultivariateNormal
warnings.filterwarnings("ignore")
import numpy as np
from seller_policy_model import PPOAgent

SELLER = 0
BUYER = 1
TRANSFORM = 2
stages = {SELLER, BUYER, TRANSFORM}

class AgentPool:
    """
    Agent pool for all the three types in the problem
    """
    def __init__(self,agents):
        self.agents = int(agents)
        self._init_agents()

    def _init_agents(self):
        """
        Initialize the seller, buyer, transformation agents 
        """
        self.seller_obs_dim = 38
        self.buyer_obs_dim = 38
        self.transer_obs_dim = 38
        self.seller_act_dim = 1
        self.buyer_act_dim = 1
        self.transer_act_dim = 1
        self.seller_cov_var = torch.full(size=(self.seller_act_dim,), fill_value=0.5)
        self.buyer_cov_var = torch.full(size=(self.buyer_act_dim,), fill_value=0.5)
        self.trans_cov_var = torch.full(size=(self.trans_act_dim,), fill_value=0.5)
        self.seller_cov_mat = torch.diag(self.seller_cov_var)
        self.buyer_cov_mat = torch.diag(self.buyer_cov_var)
        self.trans_cov_mat = torch.diag(self.trans_cov_var)
        self.agent_pools = [None for _ in range(len(stages))] 
        self.agent_pools[SELLER] = [PPOAgent(self.seller_obs_dim,\
                self.seller_act_dim) for _ in range(self.agents)]
        self.agent_pools[BUYER] = [PPOAgent(self.buyer_obs_dim,\
                self.buyer_act_dim) for _ in range(self.agents)]
        self.agent_pools[TRANSFORM] = [PPOAgent(self.transer_obs_dim,\
                self.transer_act_dim) for _ in range(self.agents)]

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
