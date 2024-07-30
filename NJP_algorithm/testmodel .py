import argparse
import torch
import time
import os
import numpy as np
import gym
from gym.wrappers import NJP
from arguments import gpsargs as args
from gym.wrappers import PredatorPreySwarmCustomizer
from gym.spaces import Box, Discrete
from torch.autograd import Variable
from algorithms.maddpg import MADDPG
from pathlib import Path
from utils.buffer import ReplayBuffer
from tensorboardX import SummaryWriter
from custom_env import MyObs, MyReward

USE_CUDA = False 

def run(config):
    model_dir = Path('./models') / config.env_id 
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]   
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = model_dir / curr_run    

    torch.manual_seed(config.seed)  
    np.random.seed(config.seed)   
    if not USE_CUDA:
        torch.set_num_threads(config.n_training_threads)  
    scenario_name = 'PredatorPreySwarm-v0'
    base_env = gym.make(scenario_name).unwrapped
    # env = NJP(base_env, args)
    custom_param = 'custom_param.json'  
    custom_param = os.path.dirname(os.path.realpath(__file__)) + '/' + custom_param
    env = NJP(base_env, custom_param)
    start_stop_num=[slice(0,env.num_predator),slice(env.num_predator, env.num_predator+env.num_prey)]               
    maddpg = MADDPG.init_from_save('./models/model_1/run5/incremental/model_ep1300.pt')

    adversary_buffer = ReplayBuffer(config.buffer_length, env.num_predator, state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0],      
                                 start_stop_index=start_stop_num[0])    
    agent_buffer = ReplayBuffer(config.buffer_length, env.num_prey,  state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0],    
                                 start_stop_index=start_stop_num[1])    
    buffer_total=[adversary_buffer, agent_buffer]        
    t = 0
    p_store = []
    h_store = []
    torch_agent_actions=[]

    explr_pct_remaining = 0.1

    print('Showing Starts...')
    print(env.penalize_control_effort)
    episode_reward = 0
    obs=env.reset()     
    maddpg.prep_rollouts(device='cpu') 
    
    maddpg.scale_noise(maddpg.noise, maddpg.epsilon)
    maddpg.reset_noise()

    M_p, N_p = np.shape(env.p)     
    M_h, N_h =np.shape(env.heading)

    p_store = np.zeros((M_p, N_p, config.episode_length))       
    h_store = np.zeros((M_h, N_h, config.episode_length))
    
    for et_i in range(config.episode_length):
        env.render()
        
        # for i, species in enumerate(num_agent):
        # Obtain observation for per agent and convert to torch variable

        p_store[:, :, et_i] = env.p             
        h_store[:, :, et_i] = env.heading

        torch_obs = torch.Tensor(obs).requires_grad_(False)  
        torch_agent_actions = maddpg.step(torch_obs, start_stop_num,  explore=True) 
        # convert actions to numpy.arrays
        agent_actions = np.column_stack([ac.data.numpy() for ac in torch_agent_actions])

        # obtain  reward and next state
        next_obs, rewards, dones, infos = env.step(agent_actions)    
        agent_buffer.push(obs, agent_actions, rewards, next_obs, dones)
        adversary_buffer.push(obs, agent_actions, rewards, next_obs, dones)  
        obs = next_obs  
        t += config.n_rollout_threads   
        episode_reward += rewards 

    maddpg.noise = max(0.05, maddpg.noise-5e-5)
    maddpg.epsilon = max(0.05, maddpg.epsilon-5e-5)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="model_1", type=str)
    parser.add_argument("--seed",
                        default=226, type=int,
                        help="Random seed")
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--buffer_length", default=int(5e5), type=int)
    parser.add_argument("--n_episodes", default=2000, type=int)
    parser.add_argument("--episode_length", default=1000, type=int)
    parser.add_argument("--batch_size",
                        default=256, type=int,
                        help="Batch size for model training")
    parser.add_argument("--n_exploration_eps", default=25000, type=int)    
    parser.add_argument("--init_noise_scale", default=0.3, type=float)       
    parser.add_argument("--final_noise_scale", default=0.0, type=float)     
    parser.add_argument("--save_interval", default=1, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--lr_actor", default=1e-4, type=float)
    parser.add_argument("--lr_critic", default=1e-3, type=float)
    parser.add_argument("--epsilon", default=0.1, type=float)
    parser.add_argument("--noise", default=0.1, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--agent_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--adversary_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])


    config = parser.parse_args()

    run(config)  
