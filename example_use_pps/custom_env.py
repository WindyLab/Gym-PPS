import gym
from gym import spaces
import numpy as np
from gym.wrappers import PredatorPreySwarmCustomizer
from gym.wrappers import CustomObservation, CustomReward, CustomAction

"""Define your own Observation and Reward in this script:
You may use the following properties to define your observation/reward functions:
self.env.p, dp, ddp, theta, heading, d_b2b_center, is_collide_b2b, energy
For detailed example, see "example_algorithm_1vs1" or "example_algorithm_NJP"
"""

class MyObs(CustomObservation):

    # def __init__(self, env, args):
    #     super().__init__(env, args)
    #     self.observation_space = spaces.Box(shape=(2, env.n_p+env.n_e), low=-np.inf, high=np.inf, dtype=np.float32)

    def observation(self, obs):
        r"""Example::

        obs = obs[6:, :]  # for example, remove ego-state
        # ⚠️ WARNING: Your algorithm should stick to your own observation space !

        """
        # your code here
        obs = obs[6:, :]
        return obs
        

class MyReward(CustomReward):
    
    def reward(self, observation, reward, action):
        r"""Example::

        reward_p =   5.0 * self.env.is_collide_b2b[self.env.n_p:self.env.n_pe, :self.env.n_p].sum(axis=0, keepdims=True).astype(float)                      
        reward_e = - 1.0 * self.env.is_collide_b2b[self.env.n_p:self.env.n_pe, :self.env.n_p].sum(axis=1, keepdims=True).astype(float).reshape(1,self.env.n_e)  
        reward_e -= 0.1 * np.abs( action[[0], self.env.n_p:self.env.n_pe]) + 0.01 * np.abs( action[[1], self.env.n_p:self.env.n_pe])  
        reward = np.concatenate((reward_p, reward_e), axis=1)

        """
        # your code here
        return reward


class MyAction(CustomAction):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.action_space = spaces.Box(shape=(2, env.n_p+env.n_e), low=-np.inf, high=np.inf)

    def action(self, action):
        # your code here
        return action
    


class MyEnv(PredatorPreySwarmCustomizer):
    def __init__(self, env, args):
        super().__init__(env, args)

    def compute_speed(self):
        speed = np.sqrt(self.env.dp[[0],:]**2 + self.env.dp[[1],:]**2)
        return speed
    
    def myfunc(self):
        # define your own function here 
        # your code here
        pass