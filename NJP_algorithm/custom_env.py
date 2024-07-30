import gym
from gym import spaces
import numpy as np

"""Define your own Observation and Reward in this script:
You may use the following properties to define your observation/reward functions:
self.env.p, dp, ddp, theta, heading, d_b2b_center, is_collide_b2b, energy
"""


class MyObs(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(shape=(55,env.n_p+env.n_e), low=-np.inf, high=np.inf)

    def observation(self, obs):
        r"""Example::

        n_pe = self.env.n_p + self.env.n_e
        obs = np.ones((2, n_pe))
        return obs

        """
        return obs
        


class MyReward(gym.RewardWrapper):

    def reward(self, reward):
        r"""Example::

        reward = np.sum(self.env.is_collide_b2b)

        """
        
        return reward

    def _get_reward(self, a):
        r"""Example::
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
                print("control_effort using MyReward")
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
        """



