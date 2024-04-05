import torch
import time
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable
from simulator import Manufacturing_Simulator

class Trainer:
    def __init__(self, num_agents=2, PPO=None, num_commodities=4, decay_factor=0.1,\
            coefs=[300,400,10,20], T=2048, history_length=32, max_timesteps_per_episode=200):
        self.num_agents = num_agents
        self.env = Manufacturing_Simulator(num_agents, num_commodities, coefs,\
                T, history_length, max_timesteps_per_episode)
        self.PPO = PPO
        self.max_timesteps_per_episode =max_timesteps_per_episode
        self.obs_dim = 38*commodities
        self.act_dim = num_commodities 
        self.num_commodities = num_commodities

    ## TODO-Revise
    def rollout(self):
        batch_obs = []             # batch observations
        batch_acts = []            # batch actions
        batch_log_probs = []       # log probs of each action
        batch_rews = []            # batch rewards
        batch_rtgs = []            # batch rewards-to-go
        batch_lens = []            # episodic lengths in batch

        # Episodic data. Keeps track of wards per episode, will get cleared
        # upon each new episode
        ep_rews = []

        t = 0 # Keeps track of how many timesteps we've run so far this batch+

        while t < self.PPO.timesteps_per_batch:
            ep_rews = []
            obs = self.reset()
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                t += 1

                # Collect observation
                batch_obs.append(obs)

                action, log_prob = self.PPO.get_action(obs)
                obs, rew, done, _ = self.step(action)

                # Collect reward, action, and log prob
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)
                if done:
                    break

                # Collect episodic length and rewards
            batch_lens.append(ep_t + 1) # plus 1 because timestep starts at 0
            batch_rews.append(ep_rews)


        # Reshape data as tensors in the shape specified before returning
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)

        # ALG STEP #4
        batch_rtgs = self.PPO.compute_rtgs(batch_rews)

        self.PPO.logger['batch_rews'] = batch_rews
        self.PPO.logger['batch_lens'] = batch_lens

        # Return the batch data
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs , batch_lens

    ## TODO-Revise
    def learn(self, total_timesteps):

        print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
        print(f"{self.PPO.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")

        t_so_far = 0 # Timesteps simulated so far
        i_so_far = 0

        while t_so_far < total_timesteps:

            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

            t_so_far += np.sum(batch_lens)

            i_so_far += 1

            self.PPO.logger['t_so_far'] = t_so_far
            self.PPO.logger['i_so_far'] = i_so_far

            for ag in range(self.num_agents):
                V, _ = self.PPO.evaluate(batch_obs[:,ag,:], batch_acts[:,ag,:],ag)
                A_k = batch_rtgs[:,ag] - V.detach()
                A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

                for _ in range(self.PPO.n_updates_per_iteration):
                    V, curr_log_probs = self.PPO.evaluate(batch_obs[:,ag,:], batch_acts[:,ag,:],ag)
                    ratios = torch.exp(curr_log_probs - batch_log_probs[:,ag])
                    surr1 = ratios * A_k
                    surr2 = torch.clamp(ratios, 1 - self.PPO.clip, 1 + self.PPO.clip) * A_k
                    actor_loss = (-torch.min(surr1, surr2)).mean()
                    critic_loss = nn.MSELoss()(V, batch_rtgs[:,ag])
                    self.PPO.actor_optim_dict[f'agent{ag+1}'].zero_grad()
                    actor_loss.backward(retain_graph=True)
                    self.PPO.actor_optim_dict[f'agent{ag+1}'].step()
                    self.PPO.critic_optim_dict[f'agent{ag+1}'].zero_grad()
                    critic_loss.backward()
                    self.PPO.critic_optim_dict[f'agent{ag+1}'].step()
                    self.PPO.logger['actor_losses'][f'agent{ag+1}'].append(actor_loss.detach())
                self.PPO._log_summary(ag)
                if i_so_far % self.PPO.save_freq == 0:
                    torch.save(self.PPO.actor_dict[f'agent{ag+1}'].state_dict(),\
                            f'./content/ppo_actor_agent{ag+1}_{i_so_far}.pth')
                    torch.save(self.PPO.critic_dict[f'agent{ag+1}'].state_dict(),\
                            f'./content/ppo_critic_agent{ag+1}_{i_so_far}.pth')
            self.PPO.logger['batch_rews'] = []
            self.PPO.logger['batch_lens'] = []
