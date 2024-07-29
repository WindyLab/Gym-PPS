'''
Specify parameters of the env
'''
from typing import Union
import numpy as np
import argparse

parser = argparse.ArgumentParser("Gym-PredatorPreySwarm Arguments")

## ==================== User settings ===================='''
parser.add_argument("--n-p", type=int, default=0, help='number of predators') 
parser.add_argument("--n-e", type=int, default=50, help='number of prey')
parser.add_argument("--is-periodic", type=bool, default=True, help='Set whether has wall or periodic boundaries') 
parser.add_argument("--dynamics-mode", type=str, default='Polar', help=" select one from ['Cartesian', 'Polar']") 
parser.add_argument("--pursuer-strategy", type=str, default='nearest', help="select one from ['input', 'static', 'random', 'nearest']") 
parser.add_argument("--escaper-strategy", type=str, default='input', help="select one from ['input', 'static', 'nearest']") 
parser.add_argument("--render-traj", type=bool, default=True, help=" whether render trajectories of agents") 
parser.add_argument("--traj_len", type=int, default=15, help="length of the trajectory") 
parser.add_argument("--billiards-mode", type=float, default=False, help="billiards mode") 
parser.add_argument("--size_p", type=float, default=0.06, help="predators size")
parser.add_argument("--size_e", type=float, default=0.035, help="evadors size")
parser.add_argument("--size_o", type=float, default=0.2, help="obstacles size")
parser.add_argument("--topo_n_p2p", type=float, default=6, help="pursuer to pursuer")
parser.add_argument("--topo_n_p2e", type=float, default=6, help="pursuer to escaper")
parser.add_argument("--topo_n_e2p", type=float, default=6, help="escaper to pursuer")
parser.add_argument("--topo_n_e2e", type=float, default=6, help="escaper to escaper")
parser.add_argument("--penalize_control_effort", type=float, default=True, help="penalize_control_effort")
parser.add_argument("--penalize_collide_walls", type=float, default=False, help="penalize_collide_walls")
parser.add_argument("--penalize_collide_agents", type=float, default=False, help="penalize_collide_agents")
parser.add_argument("--penalize_collide_obstacles", type=float, default=True, help="penalize_collide_obstacles")
## ==================== End of User settings ====================


## ==================== Advanced Settings ====================
# parser.add_argument("--action-space", type=list, default=[0, 1, 2, 3, 4] )  # up, right, down, left, stay
# parser.add_argument("--debug", type=bool, default=False )
# parser.add_argument("--animation-interval", type=float, default = 0.2)
## ==================== End of Advanced settings ====================


gpsargs = parser.parse_args()

def validate_environment_parameters(env_size, start_state, target_state, forbidden_states):
    pass
    # if not (isinstance(env_size, tuple) or isinstance(env_size, list) or isinstance(env_size, np.ndarray)) and len(env_size) != 2:
    #     raise ValueError("Invalid environment size. Expected a tuple (rows, cols) with positive dimensions.")
    
    # for i in range(2):
    #     assert start_state[i] < env_size[i]
    #     assert target_state[i] < env_size[i]
    #     for j in range(len(forbidden_states)):
    #         assert forbidden_states[j][i] < env_size[i]
# try:
#     validate_environment_parameters(gpsargs.env_size, gpsargs.start_state, args.target_state, args.forbidden_states)
# except ValueError as e:
#     print("Error:", e)