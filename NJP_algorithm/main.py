import argparse
import torch
import time
import os
import numpy as np
import gym
from gym.wrappers import NJP
from arguments import gpsargs as args
from gym.spaces import Box, Discrete
from torch.autograd import Variable
from algorithms.maddpg import MADDPG
from pathlib import Path
from utils.buffer import ReplayBuffer
from tensorboardX import SummaryWriter

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
    log_dir = run_dir / 'logs'     
    os.makedirs(log_dir)    
    logger = SummaryWriter(str(log_dir))  

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

    maddpg = MADDPG.init_from_env(env, agent_alg=config.agent_alg,
                                  adversary_alg=config.adversary_alg,
                                  tau=config.tau,
                                  lr_actor=config.lr_actor, lr_critic=config.lr_critic, epsilon=config.epsilon, noise=config.noise,
                                  hidden_dim=config.hidden_dim)             
    # maddpg = MADDPG.init_from_save('./models/model_1/run95/incremental/model_ep1000.pt')

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

    print('Training Starts...')
    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        if ep_i % 10 == 0:
            print("Episodes %i of %i" % (ep_i, config.n_episodes))
        episode_reward = 0
        obs=env.reset()     
        maddpg.prep_rollouts(device='cpu') 
      
        # explr_pct_remaining = max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps    
        # maddpg.scale_noise(config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
        maddpg.scale_noise(maddpg.noise, maddpg.epsilon)
        maddpg.reset_noise()

        M_p, N_p = np.shape(env.p)     
        M_h, N_h =np.shape(env.heading)

        p_store = np.zeros((M_p, N_p, config.episode_length))       
        h_store = np.zeros((M_h, N_h, config.episode_length))
        
        for et_i in range(config.episode_length):
            if ep_i % 50 == 0:
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

        for _ in range(30):    
            maddpg.prep_training(device='cpu')  
            for a_i in range(maddpg.nagents):
                if len(buffer_total[a_i]) >= config.batch_size:
                    sample = buffer_total[a_i].sample(config.batch_size, to_gpu=USE_CUDA)  
                    obs_sample, acs_sample, rews_sample, next_obs_sample, dones_sample = sample  
                    # assert obs_sample.size(0) == acs_sample.size(0) == rews_sample.size(0) == dones_sample.size(0)
                    maddpg.update(obs_sample, acs_sample, rews_sample, next_obs_sample, dones_sample, a_i, logger=logger)     # parameter update 
            maddpg.update_all_targets()
            maddpg.prep_rollouts(device='cpu')    
        # print("reward", episode_reward)
        DOS_epi, DOA_epi = env.dos_and_doa(x=p_store[:, start_stop_num[1], :], h=h_store[:, start_stop_num[1], :], T=config.episode_length, N=env.num_prey, D=np.sqrt(2))
        if ep_i % 10 == 0:
            print("DOS_episode:", DOS_epi, "DOA_episode:", DOA_epi)

        maddpg.noise = max(0.05, maddpg.noise-5e-5)
        maddpg.epsilon = max(0.05, maddpg.epsilon-5e-5)

        logger.add_scalar('DOS_epi', DOS_epi, global_step=ep_i)
        logger.add_scalar('DOA_epi', DOA_epi, global_step=ep_i)

        if ep_i % config.save_interval < config.n_rollout_threads:   
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            maddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            maddpg.save(run_dir / 'model.pt')

    maddpg.save(run_dir / 'model.pt')
     
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()

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
    parser.add_argument("--episode_length", default=500, type=int)
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
