import copy
import math
import torch
import time
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable
from synthetic_data_generator import synthetic_exo_data_generator, exogenous_cost_attr, exogenous_qty_attr
from torch.distributions import MultivariateNormal
from get_buyer_demands import get_buyer_quantities
seller_nonexo_state_attributes = ['current_inventory', 'action']

seller_action_attributes = ['exchange_price_action']

buyer_action_attributes = ['buyer_spot_qty', 'buyer_exchange_qty']


class Manufacturing_Simulator:
    def __init__(self, agents=2, num_commodities=4, decay_factor=0.1,\
            coefs=[300,400,10,20], T=2048, history_length=32, max_timesteps_per_episode=200):
        self.agents = agents
        self.coefs = coefs # Coefficients of the cost functions and the utility function
        self.T = T
        self.lead_time = history_length
        self.max_timesteps_per_episode =max_timesteps_per_episode
        self.obs_dim = 38*commodities
        self.act_dim = num_commodities 
        self.current_inventory_dict = {} # Key system parameters that must be updated at every time step
        self.current_obs = None
        self.exchange_price_action_dict = {} # Save the action of the agents
        self.decay_factor = decay_factor
        self.num_commodities = num_commodities

    ## TODO-Revise
    ## TODO-Split the step function into three separate functions for three steps.
    ## TODO-The collection of s,a,r for the agents will be done in the rollout function.
    def step(self, action):
        '''
        step function takes the actions to the environment as an input and calculates
        the next state observation and reward.

        ## Input: action

        prices: prices of the commodities, shape = (n_agents, n_commodities)
        waste_prices: prices of the commodity wastes, shape=(nu_agents, n_commodities)
        carrying_inventory: target rest inventory
        carrying_waste_inventory: target rest waste inventory

        Step 1: The agents decide the exchange market price, target rest inventory, waste exchange price,
            target rest waste inventory (This is the input of this function)

        Step 2: Input parameters: exchange prices, waste prices, spot price for all agents and commodities.             
        For each agent, it calls the get_buyer_quantities function to get the spot,
        exchange demand, and waste quantities by all the other agents.

        Step 3: Based on that and other exogenous attributes, it updates the inventories of the
        seller agents. Then it updates the obs state of that seller and calculates the reward.
        '''

        # Multi-commodities: the action size of each agent is equal to the number of commodities
        action = list(action)
        for i in range(len(action)):
            self.exchange_price_action_dict['agent{}'.format(i+1)] = action[i]
        ## TODO-The above is redundant
        ## Replace the self.exchange_price_action_dict by action

        seller_reward_dict = {}
        buyer_reward_dict = {}
        k = []

        #### Step 2
        # for each agent, get the spot,exchange,waste demands and also the buyer rewards of other agents.
        # Multi-commodities: get the values for each commodity 
        # Number of Commodities = |C| 
        
        # Output dimensions:
        # spot_demand_qty_agent: |N|*|C|
        # exchange_demand_qty_agent: |N|*|C|*|N|
        # waste_exchange_qty_agent: |N|*|C|*|N|
        # buyer_reward_total: |N|

        # Input dimensions:
        # prices: |N|*|C|
        # waste prices: |N|*|C|
        # inventory: |N|*|C|
        # waste inventory: |N|*|C|
        # max inventory: |N|*|C|
        # max waste inventory: |N|*|C|
        # beta: 1

        # TODO-Write a function which iterates over all the agents and collects all the results as ndarrays
        # The inputs are specified above.
        # The function should work with both baselines and DRL algorithm.
        # For the baselines such the MIP solver, the function should organize inputs, 
        # solve the optimization problem for each agent, and collect all the results.
        # For the DRL agent, the function should organize the inputs into states and generate the next states and actions. 

        # Corresponds to step2 in the draft where the sellers has decided the prices and the buyers should
        # determine the quntities.
        spot_demand_qty_agent, exchange_demand_qty_agent, waste_exchange_qty_agent, buyer_reward_total =\
                get_buyer_quantities_multi_commodities(self.exchange_price_action_dict,\
                self.agents, self.complete_dict, self.coefs, self.t)

        # TODO-Get the states for Step 3

        # TODO-Step 3
            
        for i in range(self.agents):
            # TODO-Revise the following code to prepare for everying necessary for the training.
            ## Update previous inventory
            # prev_inventory_qty: 1->M 
            prev_inventory_qty = self.current_inventory_dict[f'agent{i+1}']

            ## Update waste quantity
            # waste qty calculation => subtract the decaying factor and waste exchange demand by other agents
            self.complete_dict[f'data_dict_agent{i+1}']['waste_qty'][self.t+1] = np.max(\
                    np.zeros(self.num_commodities),self.complete_dict[f'data_dict_agent{i+1}']['waste_qty'][self.t]-\
                    np.sum(waste_exchange_qty_agent, axis=0)-\
                    self.decay_factor*self.complete_dict[f'data_dict_agent{i+1}']['waste_qty'][self.t], axis=0)
             
            ## Update current inventory
            #calculate the current inventory using => curr_invent =
            #prev_invent + produced_qty - sum(exchange demands of all agents)-
            #sum(waste exchange demands by all agents)
            self.current_inventory_dict[f'agent{i+1}'] = np.max(np.zeros(self.num_commodities),prev_inventory_qty +\
                    self.complete_dict[f'data_dict_agent{i+1}']['produced_qty'][self.t]-\
                    np.sum(exchange_demand_qty_agent, axis=0)-\
                    np.sum(waste_exchange_qty_agent, axis=0))
            
            # total demand by all agents to current seller agent.
            total_demand_agent = np.sum(spot_demand_qty_agent, axis=0) +\
                    np.sum(exchange_demand_qty_agent, axis=0) +\
                    np.sum(waste_exchange_qty_agent, axis=0)

            #update the observation state of current seller agent
            next_obs_agent = np.concatenate((\
                    self.obs_dict[f'current_obs_agent{i+1}'][1:self.lead_time],\
                    np.array([total_demand_agent]),np.array([self.current_inventory_dict[f'agent{i+1}']]),\
                    np.array([self.complete_dict[f'data_dict_agent{i+1}']['spot_price'][self.t+1]]),\
                    np.array([self.complete_dict[f'data_dict_agent{i+1}']['waste_qty'][self.t+1]]),\
                    np.array([self.complete_dict[f'data_dict_agent{i+1}']['produced_qty'][self.t+1]]),\
                    np.array([self.complete_dict[f'data_dict_agent{i+1}']['holding_cost'][self.t+1]]),\
                    np.array([self.complete_dict[f'data_dict_agent{i+1}']['seller_init_inventory']]))
            )
            #add that state to overall state observation array
            k+=[next_obs_agent]

            #calculate the seller reward for each seller agent
            seller_reward_agent = self.exchange_price_action_dict['agent{}'.format(i+1)]*\
                    min(self.current_inventory_dict[f'agent{i+1}'],\
                    sum(exchange_demand_qty_agent.values())+sum(waste_exchange_qty_agent.values()))

            for j in exchange_demand_qty_agent:
                seller_reward_agent+=self.transport_cost(exchange_demand_qty_agent[j]+waste_exchange_qty_agent[j])

            seller_reward_agent-=self.complete_dict['data_dict_agent{}'.format(i+1)]['waste_disposal_cost'][self.t]
            seller_reward_agent-=self.complete_dict['data_dict_agent{}'.format(i+1)]['holding_cost'][self.t]*\
                    self.current_inventory_dict[f'agent{i+1}']
            seller_reward_dict[f'agent{i+1}'] = seller_reward_agent

            #save the buyer rewards calculated by get_buyer_quantities in the buyer_rewards_dict.
            for buyer in buyer_reward_total:
                buyer_reward_dict[buyer] = buyer_reward_total[buyer]

        #save all the seller, buyer and total rewards.
        total_rewards = []
        seller_rewards = []
        buyer_rewards = []
        for i in range(self.agents):
            seller_rewards.append(seller_reward_dict[f'agent{i+1}'])
            buyer_rewards.append(buyer_reward_dict[f'agent{i+1}'])
            total_rewards.append(seller_reward_dict[f'agent{i+1}']+buyer_reward_dict[f'agent{i+1}'])
        
        next_obs = np.array(k)
        self.t+=1
        done = False
        if self.t > self.max_timesteps_per_episode-1 :
            done = True

        self.current_obs = next_obs

        return next_obs, np.array(seller_rewards), done, {'buyer_reward':np.array(buyer_rewards),\
                'total_reward':np.array(total_rewards)}

    ## TODO-Revise
    def reset(self):
        ## Load the parameters
        
        self.t = 0
        ## Initialize all the parameters and fill in the state values
        for i in range(self.agents):
            ## Generate exogeneous data for each agent.
            self.complete_dict[f'data_dict_agent{i+1}'] = synthetic_exo_data_generator(\
                    total_timesteps= self.T+self.lead_time, num_commodities=self.num_commodities)
            # Initialize the inventory of each commodity
            self.current_inventory_dict[f'agent{i+1}'] = [0]*self.num_commodities

            ## Fill in the state values
            self.obs_dict[f'current_obs_agent{i+1}'] = [] 
            # The first self.lead_time values of the observation are zero.
            for i in range(self.lead_time):
                self.obs_dict[f'current_obs_agent{i+1}'].append(0)
            # Add inventory numbers of each commodity to the observation
            self.obs_dict['current_obs_agent{i+1}'].append(self.current_inventory_dict[f'agent{i+1}']])
            # Add spot market price of each commnodity to the observation
            self.obs_dict['current_obs_agent{i+1}'].append(self.complete_dict[f'data_dict_agent{i+1}']['spot_price'][self.t]])
            # Add waste quantity of each commnodity to the observation
            self.obs_dict['current_obs_agent{i+1}'].append(self.complete_dict[f'data_dict_agent{i+1}']['waste_qty'][self.t]])
            self.obs_dict['current_obs_agent{i+1}'].append(self.complete_dict[f'data_dict_agent{i+1}']['produced_qty'][self.t]])
            self.obs_dict['current_obs_agent{i+1}'].append(self.complete_dict[f'data_dict_agent{i+1}']['holding_cost'][self.t]]]
            self.obs_dict['current_obs_agent{i+1}'].append(self.complete_dict[f'data_dict_agent{i+1}']['seller_init_inventory']])
        self.obs_dict[f'current_obs_agnet{i+1}'] = np.array(self.obs_dict[f'current_obs_agnet{i+1}'])

        s = []
        for i in range(self.agents):
            #s+=[self.obs_dict[f'current_obs_agent{i+1}']]
            s.append([self.obs_dict[f'current_obs_agent{i+1}']])

        self.current_obs = np.array(s)
        return self.current_obs

    def get_seller_state(self):
        """
        Get the output of the step 3 function to the states for the seller agents.
        """
        pass
    
    def get_buyer_state(self):
        """
        Get the output of the step 1 function to the states for the buyer agents.
        """
        pass
    
    def get_trans_state(self):
        """
        Get the output of the step 2 function to the states for the transformation agents.
        """
        pass
    
