import gym

class PredatorPreySwarmEnvParam(gym.Env):

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    # Agent numbers
    _n_p = 3
    _n_e = 10
    _n_o = 0

    # Environment
    _is_periodic = True
    
    # Control Strategy
    _pursuer_strategy = 'input'
    _escaper_strategy = 'input'
    _billiards_mode = False

    # Reward
    _reward_sharing_mode = 'individual'
    _penalize_control_effort = True
    _penalize_collide_walls = False         
    _penalize_distance = False
    _penalize_collide_agents = False       
    _penalize_collide_obstacles = False   

    # Metric distance for observation 
    _FoV_p = 5     # for pursuers
    _FoV_e = 5     # for escapers 

    # Topological distance for observation 
    _topo_n_p2e = 5     # pursuer to escaper 
    _topo_n_e2p = 2     # escaper to pursuer 
    _topo_n_p2p = 2     # pursuer to pursuer 
    _topo_n_e2e = 5     # escaper to escaper 

    # Action
    _act_dim_pursuer = 2
    _act_dim_escaper = 2

    # Mass
    _m_p = 3
    _m_e = 1
    _m_o = 10

    # Size
    _size_p = 0.06    
    _size_e = 0.035 
    _size_o = 0.2

    # Dynamics Mode
    _dynamics_mode = 'Polar'

    # Dynamics capability
    _linVel_p_max = 0.5  
    _linVel_e_max = 0.5
    _linAcc_p_max = 1
    _linAcc_e_max = 1
    _angle_p_max = 0.5
    _angle_e_max = 0.5

    ## Properties of obstacles
    _obstacles_cannot_move = True 
    _obstacles_is_constant = False
    if _obstacles_is_constant:   # then specify their locations:
        _p_o = np.array([[-0.5,0.5],[0,0]])

    ## Venue
    _L = 1
    _k_ball = 50       # sphere-sphere contact stiffness  N/m 
    # _c_ball = 5      # sphere-sphere contact damping N/m/s
    _k_wall = 100      # sphere-wall contact stiffness  N/m
    _c_wall = 5        # sphere-wall contact damping N/m/s
    _c_aero = 2        # sphere aerodynamic drag coefficient N/m/s

    ## Simulation Steps
    _dt = 0.1
    _n_frames = 1  
    _sensitivity = 1 

    ## Rendering
    _render_traj = True
    _traj_len = 15
    _save_frame = False


def get_param():
    params = PredatorPreySwarmEnvParam.__dict__.keys()
    params = [param for param in params if param.startswith('_') and not param.startswith('__')]
    params = [param[1:] for param in params]
    return params + ['p']

params = get_param()

    
