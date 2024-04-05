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


class PPO:
    def __init__(self,agents,hyperparameters):
        self.hyperparameters = hyperparameters
        self.agents = int(agents)
        self._init_hyperparameters()
        self._init_agents()

        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,   # timesteps so far
            'i_so_far': 0,   # iterations so far
            'batch_lens': [],# episodic lengths in batch
            'batch_rews': [],# episodic returns in batch
            'actor_losses': [[] for i in range(self.agents)],     # losses of actor network in current iteration
            "avg_batch_rews": [[] for i in range(self.agents)] ,    # avg episodic returns in batch
            "avg_actor_losses": [[] for i in range(self.agents)]    # avg losses of actor network in current iteration
        }

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
        self.agent_pools = {}
        self.agent_pools['sellers'] = [PPOAgent(self.hyperparameters, self.seller_obs_dim,\
                self.seller_act_dim) for _ in range(self.agents)]
        self.agent_pools['buyers'] = [PPOAgent(self.hyperparameters, self.buyer_obs_dim,\
                self.buyer_act_dim) for _ in range(self.agents)]
        self.agent_pools['transers'] = [PPOAgent(self.hyperparameters, self.transer_obs_dim,\
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

    ## TODO-Revise
    def compute_rtgs(self,batch_rews):

        batch_rtgs = []
        batch_shape = len(batch_rews[0])*len(batch_rews)
        # Iterate through each episode backwards to maintain same order in batch_rtgs

        for ep_rews in reversed(batch_rews):
            s  = []
            for i in range(self.agents):
              s+=[0.0]
            discounted_reward = np.array(s).reshape(batch_rews[0][0].shape) # The discounted reward so far
            ep_rtgs = []
            for rew in reversed(ep_rews):
              discounted_reward = rew + discounted_reward *self.gamma
              ep_rtgs.insert(0, discounted_reward)
            batch_rtgs.append(ep_rtgs)


        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float).reshape(batch_shape,self.agents)
        return batch_rtgs

    ## TODO-Revise
    def evaluate(self, batch_obs, batch_acts,ag):
        V = self.critic_dict[f'agent{ag+1}'](batch_obs).squeeze()
        mean = self.actor_dict[f'agent{ag+1}'](batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)
        return V, log_probs

    ## TODO-Revise
    def _log_summary(self,ag):

        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in np.array(self.logger['batch_rews'])[:,:,ag,:]])
        avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses'][f'agent{ag+1}']])
        self.logger['avg_batch_rews'][f'agent{ag+1}'].append(avg_ep_rews)
        self.logger['avg_actor_losses'][f'agent{ag+1}'].append(avg_actor_loss)
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))

        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Displaying the stats for the agent: {ag+1}", flush = True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        #self.logger['batch_lens'] = []
        #self.logger['batch_rews'] = []
        self.logger['actor_losses'][f'agent{ag+1}'] = []
