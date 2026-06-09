import gym
from gym import spaces
import numpy as np
from gym.wrappers import PredatorPreySwarmCustomizer
from gym.wrappers import CustomObservation, CustomReward, CustomAction

"""Define your own Observation and Reward in this script:
You may use the following properties to define your observation/reward functions:
self.env.p, dp, ddp, theta, heading, d_b2b_center, is_collide_b2b, energy
"""


def predator_obs(obs):
    return obs[:, 0].astype(np.float32)


def full_action(predator_action):
    action = np.zeros((2, 2), dtype=np.float32)
    action[:, 0] = predator_action
    return action


def predator_prey_distance(env):
    rel = env.p[:, 1] - env.p[:, 0]
    if env.is_periodic:
        rel[rel > env.L] -= 2 * env.L
        rel[rel < -env.L] += 2 * env.L
    return float(np.linalg.norm(rel))


class MyObs(CustomObservation):

    def __init__(self, env, args):
        super().__init__(env, args)
        self.observation_space = spaces.Box(
            shape=(env.observation_space.shape[0],),
            low=-np.inf,
            high=np.inf,
            dtype=np.float32,
        )

    def observation(self, obs):
        return predator_obs(obs)
        

class MyReward(CustomReward):
    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self._prev_distance = predator_prey_distance(self.env)
        return observation

    def reward(self, observation, reward, action):
        distance = predator_prey_distance(self.env)
        contact_distance = self.env.size_p + self.env.size_e
        caught = distance <= contact_distance
        progress = self._prev_distance - distance

        shaped = np.array(reward, copy=True, dtype=np.float32)
        shaped[0, 0] = 10.0 * float(caught) + 2.0 * progress - distance
        self._prev_distance = distance
        return shaped


class MyAction(CustomAction):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.__reinit__()

    def __reinit__(self):
        super().__reinit__()
        self.action_space = spaces.Box(shape=(2,), low=-1.0, high=1.0, dtype=np.float32)

    def action(self, action):
        return full_action(action)

    def reverse_action(self, action):
        return action[:, 0]



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
