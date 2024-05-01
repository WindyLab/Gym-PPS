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
        self.observation_space = spaces.Box(shape=(2, env.n_p+env.n_e), low=-np.inf, high=np.inf)

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



