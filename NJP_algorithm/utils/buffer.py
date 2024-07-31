import numpy as np
from torch import Tensor
from torch.autograd import Variable

class ReplayBuffer(object):
    """
    Replay Buffer for multi-agent RL with parallel rollouts
    """
    def __init__(self, max_steps, num_agents, start_stop_index, state_dim, action_dim):
        """
        Inputs:
            max_steps (int): Maximum number of timepoints to store in buffer
            num_agents (int): Number of agents in environment
            obs_dims (list of ints): number of obervation dimensions for each
                                     agent
            ac_dims (list of ints): number of action dimensions for each agent
        """
        self.max_steps = max_steps
        self.num_agents = num_agents
        self.obs_buffs = []
        self.ac_buffs = []
        self.rew_buffs = []
        self.next_obs_buffs = []
        self.done_buffs = []
        # for odim, adim in zip(obs_dims, ac_dims):     

        # self.start_number=start_index
        # self.stop_number=stop_index

        # odim1, odim2 = obs_dims[0], obs_dims[1]
        # adim1, adim2 = ac_dims[0], ac_dims[1]

        # print(odim1)
        # print(adim1)

        self.obs_buffs = np.zeros((self.max_steps * self.num_agents, state_dim)) 
        self.ac_buffs = np.zeros((self.max_steps * self.num_agents, action_dim))
        self.rew_buffs = np.zeros((self.max_steps * self.num_agents, 1))
        self.next_obs_buffs = np.zeros((self.max_steps * self.num_agents, state_dim))
        self.done_buffs = np.zeros((self.max_steps * self.num_agents, 1))

        self.filled_i = 0  # index of first empty location in buffer (last index when full)
        self.curr_i = 0  # current index to write to (ovewrite oldest data)
        # self.curr_i_obs = 0
        # self.curr_i_act = 0
        # self.curr_i_rew = 0
        # self.curr_i_next_obs = 0
        # self.curr_i_done = 0

        self.agent= start_stop_index

    def __len__(self):           
        return self.filled_i
                                            
    def push(self, observations_original, actions_original, rewards_original, next_observations_original, dones_original):
        agent_i = self.agent 

        observations = observations_original[:, agent_i].T   
        actions = actions_original[:,agent_i].T
        rewards = rewards_original[:, agent_i].T                  
        next_observations = next_observations_original[:, agent_i].T
        dones = dones_original[:, agent_i].T          

        # assert self.nentries_obs == self.nentries_next_obs == self.nentries_act == self.nentries_rew == self.nentries_done 


        
                             
        if self.curr_i + self.num_agents > self.max_steps * self.num_agents:   
            rollover = self.max_steps * self.num_agents - self.curr_i # num of indices to roll over

            self.obs_buffs = np.roll(self.obs_buffs,   
                                                rollover, axis=0)     
            self.ac_buffs = np.roll(self.ac_buffs,
                                                rollover, axis=0)
            self.rew_buffs = np.roll(self.rew_buffs,
                                                rollover, axis=0)
            self.next_obs_buffs = np.roll(self.next_obs_buffs, 
                                                    rollover, axis=0)
            self.done_buffs = np.roll(self.done_buffs,
                                                rollover, axis=0)
            self.curr_i = 0
            self.filled_i = self.max_steps
        
        self.obs_buffs[self.curr_i:self.curr_i + self.num_agents, :] = observations             
        # actions are already batched by agent, so they are indexed differently
        self.ac_buffs[self.curr_i:self.curr_i + self.num_agents, :] = actions 
        self.rew_buffs[self.curr_i:self.curr_i + self.num_agents, :] = rewards
        self.next_obs_buffs[self.curr_i:self.curr_i + self.num_agents, :] = next_observations     
        self.done_buffs[self.curr_i:self.curr_i + self.num_agents, :] = dones         

        # self.curr_i += nentries
        self.curr_i += self.num_agents

        if self.filled_i < self.max_steps:
            self.filled_i += self.num_agents         
        if self.curr_i == self.max_steps * self.num_agents: 
            self.curr_i = 0  

    def sample(self, N, to_gpu=False, norm_rews=True):
        # print("filled_i", self.filled_i)
        # print("self.max_steps * self.nentries_obs", self.max_steps * self.nentries_obs)

        inds = np.random.choice(np.arange(self.filled_i), size=N,  
                                replace=False)   

        # extracted_elements = self.obs_buffs[index_obs, :]
        
        if to_gpu:
            cast = lambda x: Tensor(x).requires_grad_(False).cuda()
        else:
            cast = lambda x: Tensor(x).requires_grad_(False)

        ret_rews = cast(self.rew_buffs[inds, :]) 
        return (cast(self.obs_buffs[inds, :]),
                cast(self.ac_buffs[inds, :]),
                ret_rews,
                cast(self.next_obs_buffs[inds, :]),
                cast(self.done_buffs[inds, :]))

    def get_average_rewards(self, N):
        if self.filled_i == self.max_steps:
            inds = np.arange(self.curr_i - N, self.curr_i)  # allow for negative indexing
        else:
            inds = np.arange(max(0, self.curr_i - N), self.curr_i)   
        return [self.rew_buffs[i][inds].mean() for i in range(self.num_agents)]
