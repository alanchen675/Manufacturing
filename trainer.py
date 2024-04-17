import torch
import time
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable
from simulator import Manufacturing_Simulator
from utility import SELLER, BUYER, TRANSFORM, stages

class Trainer:
    """
    The trainer class for the manufacturing problem

    Attributes:
    num_agents - the number of agents for each stage
    env - the environment
    agent_pool - the pool for the agents
    max_timesteps_per_episode - the maximum timesteps for every episode
    seller_obs_dim - observation size of seller agents
    buyer_obs_dim - observation size of buyer agents
    trans_obs_dim - observation size of trans agents
    seller_act_dim - action size of seller agents
    buyer_act_dim - action size of buyer agents
    trans_act_dim - action size of trans agents
    num_commodities - number of commodities

    Functoins:
    rollout - collect trajectories for training
    learn - train the agents
    """
    def __init__(self, num_agents=2, pool=None, num_commodities=4, decay_factor=0.1,\
            coefs=[300,400,10,20], T=2048, history_length=32, max_timesteps_per_episode=200):
        self.num_agents = num_agents
        self.env = Manufacturing_Simulator(num_agents, num_commodities, coefs,\
                T, history_length, max_timesteps_per_episode)
        self.agent_pool = pool 
        self.max_timesteps_per_episode =max_timesteps_per_episode
        ## TODO-Revise
        self.obs_dim = 38*commodities
        self.act_dim = num_commodities 
        self.num_commodities = num_commodities
        self._init_hyperparameters()

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
            setattr(self, param, val)
            # Sets the seed if specified
        if self.seed != None:
            # Check if our seed is valid first
            assert(type(self.seed) == int)
            # Set the seed
            torch.manual_seed(self.seed)
            print(f"Successfully set seed to {self.seed}")

    ## The three types agents make decisions in the simulator together and get their own observations
    ## Need to restore the information separately
    def rollout(self):
        # Let the following information be a list of three elements and each element can be the tensors for
        # seller, buyer, and transformation
        batch_obs = [[] for _ in len(stages)]             # batch observations. 
        batch_log_probs = [[] for _ in len(stages)]       # log probs of each action
        batch_acts = [[] for _ in len(stages)]           # batch actions
        batch_rews = [[] for _ in len(stages)]           # batch rewards
        batch_rtgs = [[] for _ in len(stages)]           # batch rewards-to-go
        batch_lens = [[] for _ in len(stages)]           # episodic lengths in batch
        # Episodic data. Keeps track of rewards per episode, will get cleared
        # upon each new episode
        ep_rews = [[] for _ in len(stages)]

        t = 0 # Keeps track of how many timesteps we've run so far this batch+

        while t < self.timesteps_per_batch:
            ep_rews = []
            obs_s = self.reset()
            # Shape: seller observations - (n_agents, seller_state_size)
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                t += 1

                # Collect seller observation
                # Append the seller_obs to the proper slot
                batch_obs[SELLER].append(obs_s)
                # Get seller action
                action_s, log_prob_s = self.agent_pool.get_action(obs_s, SELLER)
                # Send seller action and get buyer observation
                obs_b, rew_s, done_s, _ = self.step_seller(action_s)
                # Collect seller reward, action, and log prob
                ep_rews[SELLER].append(rew_s)
                batch_acts[SELLER].append(action_s)
                batch_log_probs[SELLER].append(log_prob_s)

                # Collect buyer observation
                batch_obs[BUYER].append(obs_b)
                # Get buyer action
                action_b, log_prob_b = self.agent_pool.get_action(obs_b, BUYER)
                # Send buyer action and get transformation observation
                obs_t, rew_b, done_b, _ = self.step_buyer(action_b)
                # Collect buyer reward, action, and log prob
                ep_rews[BUYER].append(rew_b)
                batch_acts[BUYER].append(action_b)
                batch_log_probs[BUYER].append(log_prob_b)

                # Collect transformtion observation
                batch_obs[TRANSFORM].append(obs_t)
                # Get transformation action
                action_t, log_prob_t = self.agent_pool.get_action(obs_t, TRANSFORM)
                # Send transformation action and get seller observation
                obs_s, rew_t, done_t, _ = self.step_buyer(action_t)
                # Collect transform reward, action, and log prob
                ep_rews[TRANSFORM].append(rew_t)
                batch_acts[TRANSFORM].append(action_t)
                batch_log_probs[TRANSFORM].append(log_prob_t)

                ## TODO-Check early termination condition
                #if done:
                #    break

                # Collect episodic length and rewards
            for stage in stages:
                batch_lens[stage].append(ep_t + 1) # plus 1 because timestep starts at 0
                batch_rews[stage].append(ep_rews[stage])


        # Reshape data as tensors in the shape specified before returning
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        #batch_obs = [torch.tensor(obs, dtype=torch.float) for obs in batch_obs]
        #batch_acts = [torch.tensor(obs, dtype=torch.float) for obs in batch_acts]
        #batch_log_probs = [torch.tensor(obs, dtype=torch.float) for obs in batch_log_probs]

        # ALG STEP #4
        batch_rtgs = self.compute_rtgs(batch_rews)

        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens

        # Return the batch data
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs , batch_lens

    def compute_rtgs(self,batch_rews):
        batch_rtgs = [[] for _ in stages] 
        for stage in stages:
            batch_rtgs[stage] = self._compute_rtgs(batch_rew[stage])
        return batch_rtgs

    def _compute_rtgs(self,batch_rews):

        batch_rtgs = []
        batch_shape = len(batch_rews[0])*len(batch_rews) # len(ep_rew)*num_episodes
        # Iterate through each episode backwards to maintain same order in batch_rtgs

        for ep_rews in reversed(batch_rews):
            s  = []
            for i in range(self.agents):
                s.append(0.0)
            discounted_reward = np.array(s).reshape(batch_rews[0][0].shape) # The discounted reward so far
            ep_rtgs = []
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward *self.gamma
                ep_rtgs.insert(0, discounted_reward)
            batch_rtgs.append(ep_rtgs)

        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float).reshape(batch_shape,self.agents)
        return batch_rtgs

    def learn(self, total_timesteps):

        #print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
        #print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")

        t_so_far = 0 # Timesteps simulated so far
        i_so_far = 0 # Batches simulated so far

        while t_so_far < total_timesteps:

            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

            t_so_far += np.sum(batch_lens)

            i_so_far += 1

            #self.logger['t_so_far'] = t_so_far
            #self.logger['i_so_far'] = i_so_far

            for ag in range(self.num_agents):
                for stage in stages:
                    self._learn(batch_obs, batch_acts, batch_log_probs, batch_rtgs, stage, ag)
                if i_so_far % self.save_freq == 0:
                    self.agent_pool.save_model(stage, ag, f'ppo_actor_agent{ag+1}_{i_so_far}.pth')
                    #torch.save(self.agent_pool.actor_dict[f'agent{ag+1}'].state_dict(),\
                    #        f'./content/ppo_actor_agent{ag+1}_{i_so_far}.pth')
                    #torch.save(self.agent_pool.critic_dict[f'agent{ag+1}'].state_dict(),\
                    #        f'./content/ppo_critic_agent{ag+1}_{i_so_far}.pth')
            self.logger['batch_rews'] = []
            self.logger['batch_lens'] = []

    ## TODO-Revise
    def _log_summary(self,ag):

        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        ## TODO-Revise
        #avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in np.array(self.logger['batch_rews'])[:,:,ag,:]])
        #avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses'][f'agent{ag+1}']])
        #self.logger['avg_batch_rews'][f'agent{ag+1}'].append(avg_ep_rews)
        #self.logger['avg_actor_losses'][f'agent{ag+1}'].append(avg_actor_loss)
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
        ## TODO-Revise
        #self.logger['actor_losses'][f'agent{ag+1}'] = []
