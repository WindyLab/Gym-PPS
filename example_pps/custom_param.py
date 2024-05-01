'''
Specify parameters of the PredatorPreySwarm environment
'''
from typing import Union
import numpy as np
import argparse

parser = argparse.ArgumentParser("Gym-PredatorPreySwarm Arguments")

parser.add_argument("--n-p", type=int, default=3, help='number of predators') 
parser.add_argument("--n-e", type=int, default=20, help='number of prey') 
parser.add_argument("--is-periodic", type=bool, default=False, help='Set whether has wall or periodic boundaries') 
parser.add_argument("--dynamics-mode", type=str, default='Polar', help=" select one from ['Cartesian', 'Polar']") 
parser.add_argument("--pursuer-strategy", type=str, default='nearest', help="select one from ['input', 'static', 'random', 'nearest']") 
parser.add_argument("--escaper-strategy", type=str, default='random', help="select one from ['input', 'static', 'random', 'nearest']") 


ppsargs = parser.parse_args()
