import os
import json
from typing import Any
import gym
import argparse
import numpy as np

class Agent:
    def __init__(self, adversary=False):
        self.adversary = adversary
        
class NJP(gym.Wrapper):

    def __init__(self, env, args):
        super(NJP, self).__init__(env)
        if isinstance(args, argparse.Namespace):
            args_ = vars(args).items()
        elif isinstance(args, dict):
            args_ = args.items()
        elif isinstance(args, str):
            print(f"Retrieving customized param from '{args}'")
            with open(args, "r") as file:
                args_ = json.load(file).items()
        else:
            raise ValueError("Invalid argument type. Parameters must be a dictionary, or argparse.Namespace, or a json file directory")
        for attr, value in args_:
            self.set_param(attr, value)
        self.__reinit__()
        print('Environment parameter customization finished.')
        self.env.n_p = args.n_p
        self.env.n_e = args.n_e
        self.env.pursuer_strategy = args.pursuer_strategy
        self.env.escaper_strategy = args.escaper_strategy
        self.env.is_periodic = args.is_periodic
        self.env.dynamics_mode = args.dynamics_mode
        self.env.render_traj = args.render_traj
        self.env.traj_len = args.traj_len
        self.env.billiards_mode = args.billiards_mode

        self.num_prey = args.n_e
        self.num_predator = args.n_p

        self.agents = [Agent() for _ in range(self.num_prey)] + [Agent(adversary=True) for _ in range(self.num_predator)]
        self.agent_types = ['adversary', 'agent']
        env.__reinit__()
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        print('NJP environment initialized successfully.')

    def set_param(self, name: str, value: Any) -> None:
        if name not in self.env.param_list:
            raise KeyError(f"Parameter '{name}' does not exist!"
            )
        setattr(self.env, name, value)
        self.__reinit__()
        

    def get_param(self, name: str) -> Any:
        if name not in self.env.param_list:
            raise KeyError(f"Parameter '{name}' does not exist!"
            )
        return getattr(self.env, name)
    
    def __reinit__(self):
        self.env.__reinit__()
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def dos_and_doa(self, x, h, T, N, D):    
        k = [0] * (self.num_prey)
        k_h = [0] * (self.num_prey)
        distances = []
        distances_h = []
        assert np.shape(x)[1] == np.shape(h)[1]
        for t in range(np.shape(x)[2]):
            for j in range(np.shape(x)[1]):
                k[j] = self._find_nearest_neighbors_DOS(x[:, :, t], j) 
                k_h[j] = self._find_nearest_neighbors_DOA(h[:, :, t], j)
                distances.append(k[j]) 
                distances_h.append(k_h[j])

        DOS = np.sum(distances) / (T * N * D)
        DOA = np.sum(distances_h) / (2 * T * N)
        return DOS, DOA
    
    def dos_and_doa_one_episode(self, x, h, N, D):
        k = [0] * (self.num_prey)
        k_h = [0] * (self.num_prey)
        distances = []
        distances_h = []
        assert np.shape(x)[1] == np.shape(h)[1]                                               
        for i in range(np.shape(x)[1]):  
            k[i] = self._find_nearest_neighbors_DOS(x, i)   
            k_h[i] = self._find_nearest_neighbors_DOA(h, i)
            distances.append(k[i]) 
            distances_h.append(k_h[i])

        DOS = np.sum(distances) / (N * D)
        DOA = np.sum(distances_h) / (2 * N)
        return DOS, DOA
    
    def _find_nearest_neighbors_DOS(self, x, i):

        distances = []
        for j in range(np.shape(x)[1]):
            if j != i:
                distances.append(np.linalg.norm(x[:, i] - x[:, j]))

        return np.min(distances)
    
    def _find_nearest_neighbors_DOA(self, x, i):

        distances = []
        for j in range(np.shape(x)[1]):
            if j != i:
                distances.append(np.linalg.norm(x[:, i] + x[:, j]))
        return np.min(distances)