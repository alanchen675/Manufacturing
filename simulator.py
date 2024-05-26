import numpy as np
from config import config, init_historical_data

class Manufacturing_Simulator:
    """
    Environment class for the manufacturing problem
    Main functions:
    -- reset: return states for the seller agents
    -- step_sell: return states for the buyer agents and rewards for the seller agents
    -- step_buy: return states for the transformation agents and rewards for the buyer agents
    -- step_trans: return states for the seller agents and rewards for the transformation agents
    The rollout stops when the step_trans function returns done=True
    """
    def __init__(self):
        """
        Initialize the environment with the configuration
        """
        for key, value in config.items():
            setattr(self, key, value)
        
    def reset(self):
        """
        Reset the environment to the initial state
        """
        # Set the starting time so the first self.history_length can  be used as historical data
        self.t = self.history_length

        ## Initilize the system parameters with the time step equal to self.history_length+self.episode_length
        ## Add additional time steps to avoid overflow
        data_length = self.history_length+self.episode_length+1
        # Shapes of the data for an episode
        general_shape = (self.num_commodities, data_length)
        individual_shape = (self.num_agents, self.num_commodities, data_length)
        pair_shape = (self.num_agents, self.num_agents, self.num_commodities, data_length)

        self.spot_price = np.zeros(shape=general_shape)
        self.price = np.zeros(shape=individual_shape)
        self.waste_price = np.zeros(shape=individual_shape)
        self.q = np.zeros(shape=pair_shape)
        self.waste_q = np.zeros(shape=pair_shape)
        self.spot_q = np.zeros(shape=individual_shape)
        self.actual_d = np.zeros(shape=pair_shape)
        self.waste_actual_d = np.zeros(shape=pair_shape)
        self.inv = np.zeros(shape=individual_shape)
        self.waste_inv = np.zeros(shape=individual_shape)
        self.inv_sell = np.zeros(shape=individual_shape)
        self.waste_inv_sell = np.zeros(shape=individual_shape)
        self.inv_buy = np.zeros(shape=individual_shape)
        self.waste_inv_buy = np.zeros(shape=individual_shape)
        self.eco_u = np.zeros(shape=individual_shape)
        self.tx_u = np.zeros(shape=individual_shape)
        self.tx_p = np.zeros(shape=individual_shape)

        historical_data = init_historical_data()
        # Make the first self.history_length time steps the same as the historical data
        for key, value in historical_data.items():
            getattr(self, key)[..., :self.history_length] = value
        ## Initialize the seller state values
        return self.get_seller_state()
    
    def get_seller_state(self):
        """
        Get the output of the step 3 function to the states for the seller agents.
        """
        seller_states = []
        start_time = self.t-self.history_length
        for n in range(self.num_agents):
            # Flatten the specific slices
            p = self.spot_price[..., start_time:self.t].flatten()
            e = self.price[..., start_time:self.t].flatten()
            ew = self.waste_price[..., start_time:self.t].flatten()
            q = self.q[..., start_time:self.t].flatten()
            qw = self.waste_q[..., start_time:self.t].flatten()
            qs = self.spot_q[n, :, start_time:self.t].flatten()
            d = self.actual_d[n, :, :, start_time:self.t].flatten()
            dw = self.waste_actual_d[n, :, :, start_time:self.t].flatten()
            I = self.inv[n,:,start_time:self.t+1].flatten()
            Iw = self.waste_inv[n,:,start_time:self.t+1].flatten()
            I_bar1 = self.inv_sell[n,:,start_time:self.t].flatten()
            Iw_bar1 = self.waste_inv_sell[n,:,start_time:self.t].flatten()
            I_bar = self.inv_buy[n,:,start_time:self.t].flatten()
            Iw_bar = self.waste_inv_buy[n,:,start_time:self.t].flatten()
            u_eco = self.eco_u[n,:,start_time:self.t].flatten()
            u_tx = self.tx_u[n,:,start_time:self.t].flatten()   
            # Concatenate flattened arrays
            state_flat = np.concatenate((p, e, ew, q, qw, qs, d, dw, I, Iw, I_bar1, Iw_bar1, I_bar, Iw_bar, u_eco, u_tx))
            
            # Append the flattened state to the list of agent states
            seller_states.append(state_flat)

        return np.array(seller_states)

    def action_conversion(self, keys, actions):
        """
        The actions input is the direct output of the RL agent.
        It has to be converted to be meaningful system values.
        conv_actions = {'price': array arr_e of shape (num_agents, num_commodities), 'waste_price': arr_ew},
        where arr_e[k] is the price of seller k for all commodities

        The function only arranges the actions input to be a dictionary now.
        The output of the RL agent are values in [0,1]. Need further conversion to make them price, etc.
        """
        # TODO - Convert the value to be meaningful system values
        conv_actions = {k: np.zeros((self.num_agents, length), dtype=actions.dtype) for k, length in keys.items()}
        for i in range(self.num_agents):
            start = 0
            for key, length in keys.items():
                conv_actions[key][i] = actions[i, start:start + length]
                start += length
        return conv_actions

    def step_sell(self, seller_states, orig_seller_actions):
        """
        Step function for the selling step
        Get the seller actions and return the states for the buyer agents
        """
        # Seller action conversion
        keys = ['price', 'waste_price', 'inv_sell', 'waste_inv_sell']
        key_len_dict = {k: self.num_commodities for k in keys}
        seller_actions = self.action_conversion(key_len_dict, orig_seller_actions)
        # Update the seller states with the seller actions
        for key, value in seller_actions.items():
            getattr(self, key)[..., self.t] = value
        # Get the buyer states and seller rewards
        buyer_states = self.get_buyer_state(keys, seller_states, seller_actions)
        seller_rewards = self.get_seller_reward()
        return buyer_states, seller_rewards
    
    def get_seller_reward(self):
        """
        Get the rewards for the seller agents
        """
        reward = (self.price[:,:,self.t]*(self.inv[:,:,self.t]-self.inv_sell[:,:,self.t])).sum(axis=1)
        return reward
    
    def get_buyer_state(self, keys, seller_states, seller_actions):
        """
        Get the output of the step 1 function to the states for the buyer agents.
        """
        buyer_states = []

        for n in range(self.num_agents):
            state_flat = seller_states[n]
            # Add the new information to the seller state of the n-th agent
            state_flat = np.concatenate((state_flat, self.spot_price[:, self.t]))
            for key in keys:
                # if 'price' in key:
                state_flat = np.concatenate((state_flat, seller_actions[key][n].flatten()))
            buyer_states.append(state_flat)

        # Update the system parameters
        for key, value in seller_actions.items():
            getattr(self, key)[..., self.t] = value
        
        return np.array(buyer_states)

    def step_buy(self, buyer_states, orig_buyer_actions):
        """
        Step function for the buying step
        Get the buyer actions and return the states for the trans agents
        """
        # Buyer action conversion
        keys = ['q', 'waste_q', 'spot_q']
        nc = self.num_agents*self.num_commodities
        lengths = [nc, nc, self.num_commodities]
        key_len_dict = {k: v for k, v in zip(keys, lengths)}
        buyer_actions = self.action_conversion(key_len_dict, orig_buyer_actions)
        for k, arr in buyer_actions.items():
            if k=='spot_q':
                continue
            buyer_actions[k] = arr.reshape(self.num_agents, self.num_agents, self.num_commodities)
        #TODO - Action conversion for buyer
        # Update the buyer states with the buyer actions
        for key, value in buyer_actions.items():
            getattr(self, key)[..., self.t] = value
        # Get trans states and buyer rewards
        trans_states = self.get_trans_state(keys, buyer_states, buyer_actions)
        buyer_rewards = self.get_buyer_reward()
        return trans_states, buyer_rewards
    
    def get_buyer_reward(self):
        """
        Get the rewards for the buyer agents
        """
        e_reshape = self.price[:,:,self.t].reshape(self.num_agents, 1, self.num_commodities)
        ew_reshape = self.waste_price[:,:,self.t].reshape(self.num_agents, 1, self.num_commodities)
        p_reshape = self.spot_price[:,self.t].reshape(self.num_commodities, 1)
        reward = -np.sum(self.actual_d[:,:,:,self.t]*e_reshape, axis=(0,2))
        reward -= np.sum(self.waste_actual_d[:,:,:,self.t]*ew_reshape, axis=(0,2))
        reward -= np.sum(self.spot_q[:,:,self.t]*p_reshape, axis=0)
        reward -= self.LAMBDA*np.sum(self.actual_d[:,:,:,self.t]-self.q[:,:,:,self.t], axis=(0,2))
        reward -= self.LAMBDA*np.sum(self.waste_actual_d[:,:,:,self.t]-self.waste_q[:,:,:,self.t], axis=(0,2))
        return reward
    
    def get_trans_state(self, keys, buyer_states, buyer_actions):
        """
        Get the output of the step 2 function to the states for the transformation agents.
        """
        actual_d = self.calc_actual_sold(self.q[:,:,:,self.t], self.inv[:,:,self.t])
        actual_dw = self.calc_actual_sold(self.waste_q[:,:,:,self.t], self.waste_inv[:,:,self.t])
        inv_buy =  self.calc_inv_buy(self.inv_sell[n,:,self.t], actual_d)
        waste_inv_buy = self.calc_inv_buy(self.waste_inv_sell[n,:,self.t], actual_dw)
        trans_states = []
        for n in range(self.num_agents):
            state_flat = buyer_states[n]
            for key in keys:
                state_flat = np.concatenate(state_flat, buyer_actions[key][n].flatten())
            # for key, value in buyer_actions.items():
            #     state_flat = np.concatenate(state_flat, value[n])

            state_flat = np.concatenate(state_flat, actual_d[n,:,:].flatten(), actual_dw[n,:,:].flatten())
            state_flat = np.concatenate(state_flat, inv_buy[n,:,:].flatten(), waste_inv_buy[n,:,:].flatten())
            trans_states.append(state_flat)
        
        # Update the system parameters
        # for key, value in buyer_actions.items():
        #     getattr(self, key)[..., self.t] = value
        self.actual_d[..., self.t] = actual_d
        self.waste_actual_d[..., self.t] = actual_dw
        self.inv_buy[..., self.t] = inv_buy
        self.waste_inv_buy[..., self.t] = waste_inv_buy
        
        return np.array(trans_states)
    
    def step_trans(self, trans_states, orig_trans_actions):
        """
        Step function for the transformation step
        Get the trans actions and return the states for the seller agents for the next time step
        """
        # Trans action conversion
        keys = ['tx_u', 'eco_u']
        key_len_dict = {k: self.num_commodities for k in keys}
        trans_actions = self.action_conversion(keys, orig_trans_actions)
        # Update the state with the trans actions
        for key, value in trans_actions.items():
            getattr(self, key)[..., self.t] = value

        # Surrogate model implementation
        # TODO-Use the real surrogate model implementation
        u_bot, w_bot = 0.1*trans_actions['tx_u'], 0.1*trans_actions['tx_u']
        # Calculate the inv and inv_waste for the next time step
        self.inv[:,:,self.t+1] = self.inv_buy[:,:,self.t]-trans_actions['tx_u'][:,:,self.t]-\
            trans_actions['eco_u'][:,:,self.t]+u_bot
        #self.waste_inv[:,:,self.t+1] = (1-self.delta)*(self.waste_inv_buy[:,:,self.t]-\
        #    trans_actions['waste_tx_u'][:,:,self.t]+w_bot)
        self.waste_inv[:,:,self.t+1] = (1-self.delta)*(self.waste_inv_buy[:,:,self.t]+w_bot)
        # Get the seller states and trans rewards
        self.t += 1
        seller_states = self.get_seller_state()
        trans_rewards = self.get_trans_reward(trans_actions)
        done = False
        # Check if the episode is done
        if self.t==self.episode_length:
            done = True
        return seller_states, trans_rewards, done
    
    def get_trans_reward(self, trans_actions):
        """
        Get the rewards for the transformation agents
        """
        reward = np.sum(trans_actions['econ_quantity'], axis=0)
        reward -= np.sum(self.tx_p[:,:,self.t]*self.tx_u[:,:,self.t], axis=1)
        return reward

    def calc_actual_sold(self, q, I):
        """
        Calculate the actual sold quantities
        """
        # Initialize the tensor d with the same shape as q
        d = np.zeros_like(q)

        # Generate the tensor d
        for c in range(self.num_commodities):
            for n in range(self.num_agents):
                # Step 1: Sort agents based on q[c, :, n] in descending order
                sorted_indices = np.argsort(-q[c, :, n])

                # Step 2: Compute d for each agent in the sorted list
                cum_sum = 0  # Initialize cumulative sum
                for i in range(self.num_agents):
                    agent_i = sorted_indices[i]
                    if i == 0:
                        # First agent n(1)
                        d[c, agent_i, n] = min(I[c, n], q[c, agent_i, n])
                    else:
                        # Subsequent agents n(i)
                        available_I = I[c, n] - cum_sum
                        d[c, agent_i, n] = min(available_I, q[c, agent_i, n])
                    
                    # Update cumulative sum
                    cum_sum += d[c, agent_i, n]

        return d
    
    def calc_inv_buy(self, I_bar, d):
        """
        Calculate the inventory bought by the buyer agents
        """
        return I_bar + np.sum(d, axis=1)
    
    def matrix_vector_product(self, q, A):
        """
        Computes the product q * A * q^T where q is a vector and A is a matrix.
        
        Parameters:
        q (np.array): A 1D numpy array of length n.
        A (np.array): A 2D numpy array of shape (n, n).
        
        Returns:
        float: The scalar result of the product q * A * q^T.

        This is for the surrogate model implementation
        """
        # Ensure q is a column vector for the multiplication
        q_row = q[np.newaxis, :]  # Make q a row vector if it isn't
        q_col = q[:, np.newaxis]  # Make q a column vector
        
        # Perform the matrix multiplication
        intermediate = np.dot(q_row, A)  # This results in a 1 x n matrix (row vector)
        result = np.dot(intermediate, q_col)  # This results in a 1 x 1 matrix (scalar)
        
        return result.item()  # Convert from 1x1 matrix to scalar
