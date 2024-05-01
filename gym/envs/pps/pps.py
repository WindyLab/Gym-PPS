__credits__ = ["lijianan@westlake.edu.cn"]

import gym
from gym import error, spaces, utils
from .putils import *
from gym.utils import *
import numpy as np
import torch
import random

class PredatorPreySwarmEnv(PredatorPreySwarmEnvProp):
    
    """
    Description:
        Multiple predators and prey interact with each other. If predators catch prey, 
        predators receive positive rewards, while prey receive negative rewards.

    Source:
        This environment appeared first in the paper Li J, Li L, Zhao S. 
        "Predator–prey survival pressure is sufficient to evolve swarming behaviors", 
        New Journal of Physics, vol. 25, no. 9, pp. 092001, 2023.

    Observation:
        Type: Box(...)
        If in Cartesian mode:
            [ agent's own pos., vel.,
              relative pos. of observed pursuers,
              relative pos. of observed escapers  ]

        If in Polar mode:
            [ agent's own pos., vel., heading,
            relative pos. and headings of observed pursuers,
            relative pos. and headings of observed escapers ]

        Observation model is dependent on both metric and topological distance.
        Metric distance: an agent can only perceive others in its perception range which is assumed to be a disk with a pre-defined radius.
        Topological distance: how many at most an agent can perceive concurrently rather than how far away.

    Actions:
        Type: Box(2)
        If the dynamics mode for agents is Cartesian, then
        Num   Action
        0     acceleration in x-axis
        1     acceleration in y-axis

        If the dynamics mode for agents is Polar, then
        Num   Action
        0     angular velocity (or rotation angle in the given time step)
        1     acceleration in heading direction

        Note: The min and max values for the action values can be adjusted, but we strongly 
        advise against doing so, as this adjustment is closely tied to the update time step. 
        Incorrectly setting these values may result in a violation of physical laws and the 
        environment dynamics may behave weirdly.

    Reward:
       The core reward is as follows: when a predator catches its prey, the predator receives 
       a reward of +1, while the prey receives a reward of -1. For details on the other 
       auxiliary rewards, please refer to the reward function.

    Starting State:
        All observations are assigned a uniform random value.

    """

    param_list = params

    def __init__(self, n_p=3, n_e=10):
        
        self._n_p = n_p
        self._n_e = n_e
        self._n_o = 0
        self.viewer = None
        self.seed()

    
    def __reinit__(self):
        self._n_pe = self._n_p + self._n_e
        self._n_peo = self._n_p + self._n_e + self._n_o
        self.observation_space = self._get_observation_space()  
        self.action_space = self._get_action_space()   
        self._m = get_mass(self._m_p, self._m_e, self._m_o, self._n_p, self._n_e, self._n_o)  
        self._size, self._sizes = get_sizes(self._size_p, self._size_e, self._size_o, self._n_p, self._n_e, self._n_o)  

        if self._billiards_mode:
            self._c_wall = 0.2
            self._c_aero = 0.02

        if self._dynamics_mode == 'Cartesian':
            self._linAcc_p_min = -1    
            self._linAcc_e_min = -1
            if self._linAcc_p_max != 1 or self._linAcc_e_max != 1:
                raise ValueError('Currently in Cartesian mode, linAcc_p_max and linAcc_e_max have to be 1')
            assert (self._linAcc_p_min, self._linAcc_e_min, self._linAcc_p_max, self._linAcc_e_max) == (-1, -1, 1, 1)
        elif self._dynamics_mode == 'Polar':
            self._linAcc_p_min = 0    
            self._linAcc_e_min = 0
            
        # Energy
        if self._dynamics_mode == 'Cartesian':
            self.max_energy_p = 1000. 
            self.max_energy_e = 1000.  
        elif self._dynamics_mode == 'Polar':
            self.max_energy_p = 1000. 
            self.max_energy_e = 1000.  


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def reset(self):
        max_size = np.max(self._size)
        max_respawn_times = 100
        for respawn_time in range (max_respawn_times):
            self._p = np.random.uniform(-1+max_size, 1-max_size, (2, self._n_peo))   # Initialize self._p
            if self._obstacles_is_constant:
                self._p[:, self._n_pe:self._n_peo] = self._p_o
            self._d_b2b_center, _, _is_collide_b2b = get_dist_b2b(self._p, self._L, self._is_periodic, self._sizes)
            if _is_collide_b2b.sum() == 0:
                break
            if respawn_time == max_respawn_times-1:
                print('Some particles are overlapped at the initial time !')
        if self._render_traj == True:
            self._p_traj = np.zeros((self._traj_len, 2, self._n_peo))
            self._p_traj[0,:,:] = self._p
        self._dp = np.zeros((2, self._n_peo))  
        if self._billiards_mode:
            self._dp = np.random.uniform(-1,1,(2,self._n_peo))  # ice mode  
            if self._dynamics_mode == 'Polar':
                raise ValueError("Billiards_mode requires dynamics_mode be 'Cartesian' !")                     
        if self._obstacles_cannot_move:
            self._dp[:, self._n_pe:self._n_peo] = 0
        self._ddp = np.zeros((2, self._n_peo))                                     
        self._energy = np.array([self.max_energy_p for _ in range(self._n_p)] + [self.max_energy_e for _ in range(self.n_e)]).reshape(1, self._n_pe)
        if self._dynamics_mode == 'Polar': 
            self._theta = np.pi * np.random.uniform(-1,1, (1, self._n_peo))
            # self._theta = np.pi * np.zeros((1, self._n_peo))
            self._heading = np.concatenate((np.cos(self._theta), np.sin(self._theta)), axis=0)  
        return self._get_obs()


    def _get_obs(self):

        self.obs = np.zeros(self.observation_space.shape)   

        for i in range(self._n_p):
            ''' For pursuers

            If in Cartesian mode:
            [ agent's own pos., vel.,
              relative pos. of observed pursuers,
              relative pos. of observed escapers  ]

            If in Polar mode:
            [ agent's own pos., vel., heading,
              relative pos. and headings of observed pursuers,
              relative pos. and headings of observed escapers ]

            Observation model is dependent on both metric and topological distance.
            Metric distance means: an agent can only perceive others in its perception range which is assumed to be a disk with a pre-defined radius.
            Topological distancemeans: how many at most an agent can perceive concurrently rather than how far away.
            '''
            relPos_p2p = self._p[:, :self._n_p] - self._p[:,[i]]
            if self._is_periodic: relPos_p2p = make_periodic( relPos_p2p, self._L )
            relVel_p2p = self._dp[:,:self._n_p] - self._dp[:,[i]] if self._dynamics_mode == 'Cartesian' else self._heading[:, :self._n_p] - self._heading[:, [i]]
            relPos_p2p, relVel_p2p = get_focused(relPos_p2p, relVel_p2p, self._FoV_p, self._topo_n_p2p, True)  
           
            relPos_p2e = self._p[:, self._n_p:self._n_pe] - self._p[:,[i]]
            if self._is_periodic: relPos_p2e = make_periodic( relPos_p2e, self._L )
            relVel_p2e = self._dp[:,self._n_p:self._n_pe] - self._dp[:,[i]] if self._dynamics_mode == 'Cartesian' else self._heading[:,self._n_p:self._n_pe] - self._heading[:,[i]]
            relPos_p2e, relVel_p2e = get_focused(relPos_p2e, relVel_p2e, self._FoV_p, self._topo_n_p2e, False) 
          
            obs_pursuer_pos = np.concatenate((self._p[:, [i]], relPos_p2p, relPos_p2e), axis=1)
            obs_pursuer_vel = np.concatenate((self._dp[:, [i]], relVel_p2p, relVel_p2e), axis=1)
            obs_pursuer = np.concatenate((obs_pursuer_pos, obs_pursuer_vel), axis=0)  # (4, n_peo+1) FIXME: only suitable for no obstacles

            if self._dynamics_mode == 'Cartesian':
                self.obs[:self.obs_dim_pursuer-1, i] = obs_pursuer.T.reshape(-1)       
                self.obs[self.obs_dim_pursuer-1, i] = 2/self.max_energy_p * self._energy[[0],i] - 1  
            elif self._dynamics_mode == 'Polar':
                self.obs[:self.obs_dim_pursuer-3, i] = obs_pursuer.T.reshape(-1)       
                self.obs[self.obs_dim_pursuer-3, i] = 2/self.max_energy_p * self._energy[[0],i] - 1  
                self.obs[self.obs_dim_pursuer-2:self.obs_dim_pursuer, i] = self._heading[:,i]

        for i in range(self._n_p, self._n_pe):
            ''' For prey
            Same with predators'
            '''
            relPos_e2p = self._p[:, :self._n_p] - self._p[:,[i]]  
            if self._is_periodic: relPos_e2p = make_periodic( relPos_e2p, self._L )
            relVel_e2p = self._dp[:, :self._n_p] - self._dp[:,[i]] if self._dynamics_mode == 'Cartesian' else self._heading[:, :self._n_p] - self._heading[:,[i]]
            relPos_e2p, relVel_e2p = get_focused(relPos_e2p, relVel_e2p, self._FoV_e, self._topo_n_e2p, False) 
            
            relPos_e2e = self._p[:, self._n_p:self._n_pe] - self._p[:,[i]]
            if self._is_periodic: relPos_e2e = make_periodic( relPos_e2e, self._L )
            relVel_e2e = self._dp[:, self._n_p:self._n_pe] - self._dp[:,[i]] if self._dynamics_mode == 'Cartesian' else self._heading[:, self._n_p:self._n_pe] - self._heading[:,[i]]
            relPos_e2e, relVel_e2e = get_focused(relPos_e2e, relVel_e2e, self._FoV_e, self._topo_n_e2e, True)  

            obs_escaper_pos = np.concatenate((self._p[:, [i]], relPos_e2p, relPos_e2e), axis=1)
            obs_escaper_vel = np.concatenate((self._dp[:, [i]], relVel_e2p, relVel_e2e), axis=1)
            obs_escaper = np.concatenate((obs_escaper_pos, obs_escaper_vel), axis=0) 

            if self._dynamics_mode == 'Cartesian':
                self.obs[:self.obs_dim_escaper-1, i] = obs_escaper.T.reshape(-1)       
                self.obs[self.obs_dim_escaper-1,i] = 2/self.max_energy_e * self._energy[[0],i] - 1   
            elif self._dynamics_mode == 'Polar':
                self.obs[:self.obs_dim_escaper-3, i] = obs_escaper.T.reshape(-1)       
                self.obs[self.obs_dim_escaper-3, i] = 2/self.max_energy_e * self._energy[[0],i] - 1  
                self.obs[self.obs_dim_escaper-2:self.obs_dim_escaper, i] = self._heading[:,i] 

        return self.obs
      

    def _get_reward(self, a):

        reward_p =   5.0 * self._is_collide_b2b[self._n_p:self._n_pe, :self._n_p].sum(axis=0, keepdims=True).astype(float)                      
        reward_e = - 5.0 * self._is_collide_b2b[self._n_p:self._n_pe, :self._n_p].sum(axis=1, keepdims=True).astype(float).reshape(1,self.n_e)  

        if self._penalize_distance:
            reward_p += - self._d_b2b_center[self._n_p:self._n_pe, :self._n_p].sum(axis=0, keepdims=True)
            reward_e +=   self._d_b2b_center[self._n_p:self._n_pe, :self._n_p].sum(axis=1, keepdims=True).reshape(1,self.n_e)

        if self._penalize_control_effort:
            if self._dynamics_mode == 'Cartesian':
                reward_p -= 1*np.sqrt( a[[0],:self._n_p]**2 + a[[1],:self._n_p]**2 )
                reward_e -= 1*np.sqrt( a[[0], self._n_p:self._n_pe]**2 + a[[1], self._n_p:self._n_pe]**2 )
            elif self._dynamics_mode == 'Polar':
                reward_p -= 1 * np.abs( a[[0], :self._n_p] ) +         0 * np.abs( a[[1], :self._n_p] )
                reward_e -= 1 * np.abs( a[[0], self._n_p:self._n_pe]) + 0 * np.abs( a[[1], self._n_p:self._n_pe])     
      
        if self._penalize_collide_agents:
            reward_p -= self._is_collide_b2b[:self._n_p, :self._n_p].sum(axis=0, keepdims=True)
            reward_e -= self._is_collide_b2b[self._n_p:self._n_pe, self._n_p:self._n_pe].sum(axis=0, keepdims=True)

        if self._penalize_collide_obstacles:
            reward_p -= 5 * self._is_collide_b2b[self._n_pe:self._n_peo, 0:self._n_p].sum(axis=0, keepdims=True)          
            reward_e -= 5 * self._is_collide_b2b[self._n_pe:self._n_peo, self._n_p:self._n_pe].sum(axis=0, keepdims=True) 
        
        if self._penalize_collide_walls and self._is_periodic == False:
            reward_p -= 1 * self.is_collide_b2w[:, :self._n_p].sum(axis=0, keepdims=True)            
            reward_e -= 1 * self.is_collide_b2w[:, self._n_p:self._n_pe].sum(axis=0, keepdims=True)  

        if self._reward_sharing_mode == 'sharing_mean':
            reward_p[:] = np.mean(reward_p) 
            reward_e[:] = np.mean(reward_e)
        elif self._reward_sharing_mode == 'sharing_max':
            reward_p[:] = np.max(reward_p) 
            reward_e[:] = np.max(reward_e)
        elif self._reward_sharing_mode == 'individual':
            pass
        else:
            print('reward mode error !!')

        reward = np.concatenate((reward_p, reward_e), axis=1) 
        return reward


    def _get_done(self):
        # all_done = np.zeros( (1, self._n_pe) ).astype(bool)
        # return all_done
        return False


    def _get_info(self):
        dist_matrix = self._d_b2b_center[self._n_p:self._n_pe, self._n_p:self._n_pe]
        dist_matrix += 10 * np.identity(self.n_e)   
        ave_min_dist = np.mean( np.min(dist_matrix, axis=0) )
        DoC = 1/ave_min_dist

        nearest_idx = np.argmin(dist_matrix, axis=0)
        if self._dynamics_mode == 'Cartesian':
            nearest_headings = self._dp[:, self._n_p:self._n_pe][:, nearest_idx]
        elif self._dynamics_mode == 'Polar':
            nearest_headings = self._heading[:, self._n_p:self._n_pe][:, nearest_idx]
            
        # alignments = self._heading[:, self._n_p:self._n_pe] + nearest_headings
        # DoA = np.mean( np.sqrt( alignments[0,:]**2 + alignments[1,:]**2 ) )
        

        # TODO
        # assert self.n_e >= 2
        # ave_dist =   self._d_b2b_center[self._n_p:self._n_pe, self._n_p:self._n_pe].sum() / self.n_e / (self.n_e-1)
        # DoC_gloal = 1/ave_dist
        DoC_gloal = 0

        # return np.array( [DoC, DoA, DoC_gloal] ).reshape(3,1)
        return np.array( [None, None, None] ).reshape(3,1)


    def step(self, a):  
        for _ in range(self._n_frames): 
            if self._dynamics_mode == 'Polar':  
                a[0, :self._n_p] *= self._angle_p_max
                a[0, self._n_p:self._n_pe] *= self._angle_e_max
                a[1, :self._n_p] =          (self._linAcc_p_max-self._linAcc_p_min)/2 * a[1,:self._n_p] +          (self._linAcc_p_max+self._linAcc_p_min)/2 
                a[1, self._n_p:self._n_pe] = (self._linAcc_e_max-self._linAcc_e_min)/2 * a[1,self._n_p:self._n_pe] + (self._linAcc_e_max+self._linAcc_e_min)/2 

            # self._d_b2b_center, self.d_b2b_edge, self._is_collide_b2b = self._get_dist_b2b()
            self._d_b2b_center, self.d_b2b_edge, self._is_collide_b2b = get_dist_b2b(self._p, self._L, self._is_periodic, self._sizes)
            sf_b2b_all = np.zeros((2*self._n_peo, self._n_peo)) 
            for i in range(self._n_peo):
                for j in range(i):
                    delta = self._p[:,j]-self._p[:,i]
                    if self._is_periodic:
                        delta = make_periodic(delta, self._L)
                    dir = delta / self._d_b2b_center[i,j]
                    sf_b2b_all[2*i:2*(i+1),j] = self._is_collide_b2b[i,j] * self.d_b2b_edge[i,j] * self._k_ball * (-dir)
                    sf_b2b_all[2*j:2*(j+1),i] = - sf_b2b_all[2*i:2*(i+1),j]  
                   
            sf_b2b = np.sum(sf_b2b_all, axis=1, keepdims=True).reshape(self._n_peo,2).T   

            if self._is_periodic == False:
                # self.d_b2w, self.is_collide_b2w = self._get_dist_b2w()
                self.d_b2w, self.is_collide_b2w = get_dist_b2w(self._p, self._size, self._L)
                sf_b2w = np.array([[1, 0, -1, 0], [0, -1, 0, 1]]).dot(self.is_collide_b2w * self.d_b2w) * self._k_wall   
                df_b2w = np.array([[-1, 0, -1, 0], [0, -1, 0, -1]]).dot(self.is_collide_b2w*np.concatenate((self._dp, self._dp), axis=0))  *  self._c_wall   

            if self.pursuer_strategy == 'input':
                pass
            elif self.pursuer_strategy == 'static':
                a[:,:self._n_p] = np.zeros((self._act_dim_pursuer, self._n_p))                
            elif self.pursuer_strategy == 'random':
                a[:,:self._n_p] = np.random.uniform(-1,1, (self._act_dim_pursuer, self._n_p)) 
                if self._dynamics_mode == 'Polar': 
                    a[0, :self._n_p] *= self._angle_p_max
                    a[1, :self._n_p] = (self._linAcc_p_max-self._linAcc_p_min)/2 * a[1,:self._n_p] + (self._linAcc_p_max+self._linAcc_p_min)/2 
            elif self.pursuer_strategy == 'nearest':
                ind_nearest = np.argmin( self._d_b2b_center[:self._n_p, self._n_p:self._n_pe], axis=1)
                goto_pos =  self._p[:, self._n_p+ind_nearest] - self._p[:,:self._n_p]   
                if self._is_periodic == True:
                    goto_pos = make_periodic( goto_pos, self._L )
                ranges = np.sqrt( goto_pos[[0],:]**2 + goto_pos[[1],:]**2 )
                goto_dir = goto_pos / ranges   
                if self._dynamics_mode == 'Cartesian':
                    a[:,:self._n_p] = 1 * goto_dir
                elif self._dynamics_mode == 'Polar':
                    goto_dir = np.concatenate( (goto_dir, np.zeros((1,self._n_p))), axis=0 ).T 
                    heading = np.concatenate( (self._heading[:,:self._n_p], np.zeros((1, self._n_p))), axis=0 ).T
                    desired_rotate_angle = np.cross(heading, goto_dir)[:,-1] 
                    desired_rotate_angle[desired_rotate_angle>self._angle_p_max] = self._angle_p_max
                    desired_rotate_angle[desired_rotate_angle<-self._angle_p_max] = -self._angle_p_max
                    a[0, :self._n_p] = desired_rotate_angle
                    a[1, :self._n_p] = self._linAcc_p_max
            else:
                print('Wrong in Step function')
                    
            
            if self.escaper_strategy == 'input':
                pass
            elif self.escaper_strategy == 'static':
                a[:,self._n_p:self._n_pe] = np.zeros((self._act_dim_escaper, self.n_e))
            elif self.escaper_strategy == 'random':
                a[:,self._n_p:self._n_pe] = np.random.uniform(-1,1, (self._act_dim_escaper, self.n_e)) 
                if self._dynamics_mode == 'Polar': 
                    a[0, self._n_p:self._n_pe] *= self._angle_e_max
                    a[1, self._n_p:self._n_pe] = (self._linAcc_e_max-self._linAcc_e_min)/2 * a[1,self._n_p:self._n_pe] + (self._linAcc_e_max+self._linAcc_e_min)/2 
            elif self.escaper_strategy == 'nearest':
                ind_nearest = np.argmin( self._d_b2b_center[self._n_p:self._n_pe, :self._n_p], axis=1)
                goto_pos =  - self._p[:, ind_nearest] + self._p[:, self._n_p:self._n_pe]  
                if self._is_periodic == True:
                    goto_pos = make_periodic( goto_pos, self._L )
                ranges = np.sqrt( goto_pos[[0],:]**2 + goto_pos[[1],:]**2 )
                goto_dir = goto_pos / ranges  
                if self._dynamics_mode == 'Cartesian':
                    a[:, self._n_p:self._n_pe] = 1 * goto_dir
                elif self._dynamics_mode == 'Polar':
                    goto_dir = np.concatenate( (goto_dir, np.zeros((1,self.n_e))), axis=0 ).T 
                    heading = np.concatenate( (self._heading[:,self._n_p:self._n_pe], np.zeros((1, self.n_e))), axis=0 ).T 
                    desired_rotate_angle = np.cross(heading, goto_dir)[:,-1]  
                    desired_rotate_angle[desired_rotate_angle>self._angle_e_max] = self._angle_e_max
                    desired_rotate_angle[desired_rotate_angle<-self._angle_e_max] = -self._angle_e_max
                    a[0, self._n_p:self._n_pe] = desired_rotate_angle
                    a[1, self._n_p:self._n_pe] = self._linAcc_e_max 
            else:
                print('Wrong in Step function')    


            if self._dynamics_mode == 'Cartesian':
                u = a   
            elif self._dynamics_mode == 'Polar': 
                self._theta += a[[0],:]
                self._theta = normalize_angle(self._theta)
                self._heading = np.concatenate((np.cos(self._theta), np.sin(self._theta)), axis=0) 
                u = a[[1], :] * self._heading 
            else:
                print('Wrong in updating dynamics')

            if self._is_periodic == True:
                F = self._sensitivity * u  + sf_b2b - self._c_aero*self._dp
                # F = self._sensitivity * u  + sf_b2b + df_b2b - self._c_aero*dp
            elif self._is_periodic == False:
                F = self._sensitivity * u  + sf_b2b - self._c_aero*self._dp + sf_b2w + df_b2w 
            else:
                print('Wrong in considering walls !!!')
            self._ddp = F/self._m
            self._dp += self._ddp * self._dt
            if self._obstacles_cannot_move:
                self._dp[:, self._n_pe:self._n_peo] = 0
            self._dp[:,:self._n_p] = np.clip(self._dp[:,:self._n_p], -self._linVel_p_max, self._linVel_p_max)
            self._dp[:,self._n_p:self._n_pe] = np.clip(self._dp[:,self._n_p:self._n_pe], -self._linVel_e_max, self._linVel_e_max)
            energy = np.tile(self._energy, (2,1))
            self._dp[:,:self._n_pe][energy<0.5] = 0
            speeds = np.sqrt( self._dp[[0],:self._n_pe]**2 + self._dp[[1],:self._n_pe]**2 )
            self._energy -= speeds
            self._energy[speeds<0.01] += 0.1   
            self._energy[0,:self._n_p][self._energy[0,:self._n_p]>self.max_energy_p] = self.max_energy_p
            self._energy[0,self._n_p:][self._energy[0,self._n_p:]>self.max_energy_e] = self.max_energy_e
            self._p += self._dp * self._dt
            if self._obstacles_is_constant:
                self._p[:, self._n_pe:self._n_peo] = self._p_o
            if self._is_periodic:
                self._p = make_periodic(self._p, self._L)


            if self._render_traj == True:
                self._p_traj = np.concatenate( (self._p_traj[1:,:,:], self._p.reshape(1, 2, self._n_peo)), axis=0 )

        return self._get_obs(), self._get_reward(a), self._get_done(), self._get_info()

        # TODO: obstacle or shelter
    # ============== ================= =====================


    def render(self, mode="human"): 
    
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-1, 1, -1, 1.)
            
            agents = []
            self.tf = []
            if self._render_traj: self.trajrender = []
            for i in range(self._n_pe):
                if self._render_traj: self.trajrender.append( rendering.Traj( list(zip(self._p_traj[:,0,i], self._p_traj[:,1,i])),  False) )
                if i < self._n_p:
                    if self._dynamics_mode == 'Polar':
                        agents.append( rendering.make_unicycle(self._size_p) )
                    elif self._dynamics_mode == 'Cartesian':
                        agents.append( rendering.make_circle(self._size_p) )
                    agents[i].set_color_alpha(1, 0.5, 0, 1)
                    if self._render_traj: self.trajrender[i].set_color_alpha(1, 0.5, 0, 0.5)
                elif (i >=self._n_p) and (i<self._n_pe):
                    if self._dynamics_mode == 'Polar':
                        agents.append( rendering.make_unicycle(self._size_e) )
                    elif self._dynamics_mode == 'Cartesian':
                        agents.append( rendering.make_circle(self._size_e) )
                    agents[i].set_color_alpha(0, 0.333, 0.778, 1)
                    if self._render_traj: self.trajrender[i].set_color_alpha(0, 0.333, 0.778, 0.5)
                self.tf.append( rendering.Transform() )
                agents[i].add_attr(self.tf[i])
                self.viewer.add_geom(agents[i])
                if self._render_traj: self.viewer.add_geom(self.trajrender[i])

        for i in range(self._n_pe):
            if self._dynamics_mode == 'Polar':
                self.tf[i].set_rotation(self._theta[0,i])
            elif self._dynamics_mode == 'Cartesian':
                pass
            self.tf[i].set_translation(self._p[0,i], self._p[1,i])
            if self._render_traj: self.trajrender[i].set_traj(list(zip(self._p_traj[:,0,i], self._p_traj[:,1,i])))

        return self.viewer.render(return_rgb_array=mode == "rgb_array")
    
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
    
    def _get_observation_space(self):
        self._topo_n_p = self._topo_n_p2p + self._topo_n_p2e
        self._topo_n_e = self._topo_n_e2p + self._topo_n_e2e 
        self.obs_dim_pursuer = ( 2 + 2*self._topo_n_p ) * 2  + 1    
        self.obs_dim_escaper = ( 2 + 2*self._topo_n_e ) * 2  + 1   
        if self._dynamics_mode == 'Polar':
            self.obs_dim_pursuer += 2
            self.obs_dim_escaper += 2  
        obs_dim_max = np.max([self.obs_dim_pursuer, self.obs_dim_escaper])   
        observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim_max, self._n_pe), dtype=np.float32)
        return observation_space

    def _get_action_space(self):
        _act_dim_max = np.max([self._act_dim_pursuer, self._act_dim_escaper])
        action_space = spaces.Box(low=-1, high=1, shape=(_act_dim_max, self._n_pe), dtype=np.float32)
        return action_space


if __name__ == '__main__':

    env = PredatorPreySwarmEnv()
    Pos = np.array([ [1, 2, 3, 0, 1],
                     [2, 3, 4, 2, 2.3] ])
    Vel = np.array([ [1, 2, 3, 4, 5],
                     [1, 2, 3, 4, 5] ])
    print(Pos)
    print(Vel)
    threshold = 5
    desired_n = 2
    get_focused(Pos, Vel, threshold, desired_n, False)
