import torch
import torch.nn.functional as F
from gym.spaces import Box, Discrete
from utils.networks import MLPNetwork
from utils.misc import soft_update, average_gradients, onehot_from_logits, gumbel_softmax
from utils.agents import DDPGAgent

MSELoss = torch.nn.MSELoss()

class MADDPG(object):
    """
    Wrapper class for DDPG-esque (i.e. also MADDPG) agents in multi-agent task
    """
    def __init__(self, agent_init_params, alg_types, epsilon, noise,
                 gamma=0.95, tau=0.01, lr_actor=1e-4, lr_critic=1e-3,  hidden_dim=64, 
                 discrete_action=False):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
                num_in_critic (int): Input dimensions to critic
            alg_types (list of str): Learning algorithm for each agent (DDPG
                                       or MADDPG)
            gamma (float): Discount factor
            tau (float): Target update rate
            lr (float): Learning rate for policy and critic
            hidden_dim (int): Number of hidden dimensions for networks
            discrete_action (bool): Whether or not to use discrete action space
        """
        self.nagents = len(alg_types)
        self.alg_types = alg_types
        self.epsilon = epsilon
        self.noise = noise
        self.agents = [DDPGAgent(lr_actor=lr_actor, lr_critic=lr_critic, discrete_action=discrete_action,   # 每个 agent 除了 agent inital parameters 不一样之外，都是同构的
                                 hidden_dim=hidden_dim, epsilon=self.epsilon, noise=self.noise,
                                 **params)   
                       for params in agent_init_params]   
        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.discrete_action = discrete_action
        self.pol_dev = 'cpu'  
        self.critic_dev = 'cpu' 
        self.trgt_pol_dev = 'cpu' 
        self.trgt_critic_dev = 'cpu'  
        self.niter = 0

    @property           
    def policies(self):
        return [a.policy for a in self.agents]

    
    def target_policies(self, agent_i, obs):
        return self.agents[agent_i].target_policy(obs)

    def scale_noise(self, scale, new_epsilon):
        """
        Scale noise for each agent
        Inputs:
            scale (float): scale of noise
        """
        for a in self.agents:
            a.scale_noise(scale)          # 这个没有 return，目的是修改类里面的属性值
            a.epsilon = new_epsilon

    def reset_noise(self):
        for a in self.agents:
            a.reset_noise()

    def step(self, observations, start_stop_num, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """                                                           
        return [self.agents[i].step(observations[:, start_stop_num[i]].t(), explore=explore) for i in range(len(start_stop_num))]

    def update(self, obs, acs, rews, next_obs, dones, agent_i, parallel=False, logger=None):
        """
        Update parameters of agent model based on sample from replay buffer
        Inputs:
            sample: tuple of (observations, actions, rewards, next
                    observations, and episode end masks) sampled randomly from
                    the replay buffer. Each is a list with entries
                    corresponding to each agent
            agent_i (int): index of agent to update
            parallel (bool): If true, will average gradients across threads
            logger (SummaryWriter from Tensorboard-Pytorch):
                If passed in, important quantities will be logged
        """
        # obs, acs, rews, next_obs, dones = sample            
        curr_agent = self.agents[agent_i]           
        curr_agent.critic_optimizer.zero_grad()     
        all_trgt_acs = self.target_policies(agent_i, next_obs)  
        trgt_vf_in = torch.cat((next_obs, all_trgt_acs), dim=1)  
        target_value = (rews + self.gamma *
                        curr_agent.target_critic(trgt_vf_in) *  
                        (1 - dones))                                               
        vf_in = torch.cat((obs, acs), dim=1)
        actual_value = curr_agent.critic(vf_in)           
        vf_loss = MSELoss(actual_value, target_value.detach()) 
        # vf_loss = (actual_value-target_value.detach()) ** 2
        vf_loss.backward()
        if parallel:
            average_gradients(curr_agent.critic)
        # torch.nn.utils.clip_grad_norm(curr_agent.critic.parameters(), 0.5)
        curr_agent.critic_optimizer.step()
        curr_agent.policy_optimizer.zero_grad()   

        if not self.discrete_action:
            curr_pol_out = curr_agent.policy(obs)
            curr_pol_vf_in = curr_pol_out

        all_pol_acs = curr_pol_vf_in  
        vf_in = torch.cat((obs, all_pol_acs), dim=1)

        pol_loss = -curr_agent.critic(vf_in).mean()      
        pol_loss.backward()     
                                
        if parallel:
            average_gradients(curr_agent.policy)
        # torch.nn.utils.clip_grad_norm(curr_agent.policy.parameters(), 0.5)   
        curr_agent.policy_optimizer.step() 
        if logger is not None:
            logger.add_scalars('agent%i/losses' % agent_i,
                               {'vf_loss': vf_loss,
                                'pol_loss': pol_loss},
                               self.niter)

    def update_all_targets(self):    
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        for a in self.agents:
            soft_update(a.target_critic, a.critic, self.tau)   
            soft_update(a.target_policy, a.policy, self.tau)
        self.niter += 1

    def prep_training(self, device='gpu'):
        for a in self.agents:
            a.policy.train()  
            a.target_policy.train()
            a.target_critic.train()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device
        if not self.critic_dev == device:
            for a in self.agents:
                a.critic = fn(a.critic)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_policy = fn(a.target_policy)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            for a in self.agents:
                a.target_critic = fn(a.target_critic)
            self.trgt_critic_dev = device

    def prep_rollouts(self, device='cpu'):
        for a in self.agents:
            a.policy.eval()   
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [a.get_params() for a in self.agents]}
        torch.save(save_dict, filename)

   
    @classmethod      
    def init_from_env(cls, env, agent_alg="MADDPG", adversary_alg="MADDPG",
                      gamma=0.95, tau=0.01, lr_actor=1e-4, lr_critic=1e-3, hidden_dim=64, epsilon=0.1, noise=0.1):
        """
        Instantiate instance of this class from multi-agent environment
        """
        agent_init_params = []
        num_in_pol=env.observation_space.shape[0]
        num_out_pol=env.action_space.shape[0]
        num_in_critic=env.observation_space.shape[0] + env.action_space.shape[0]

        # print("num in pol", num_in_pol, "num out pol", num_out_pol, "num in critic", num_in_critic)

        alg_types = [adversary_alg if atype == 'adversary' else agent_alg for
                     atype in env.agent_types]   
                     
        for algtype in alg_types:  
            agent_init_params.append({'num_in_pol': num_in_pol,
                                      'num_out_pol': num_out_pol,
                                      'num_in_critic': num_in_critic})
        init_dict = {'gamma': gamma, 'tau': tau, 'lr_actor': lr_actor, 'lr_critic': lr_critic, 'epsilon': epsilon, 'noise': noise,
                     'hidden_dim': hidden_dim,
                     'alg_types': alg_types,
                     'agent_init_params': agent_init_params}
        instance = cls(**init_dict)    
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename):    
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)
        return instance