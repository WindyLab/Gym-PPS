import gym
from .param import *
import numpy as np

class PredatorPreySwarmEnvProp(PredatorPreySwarmEnvParam):
    
    ## Useful parameters to customize observations and reward functions

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, value):
        self._assert_2X_ndarray('p', value)
        self._p = value

    @property
    def dp(self):
        return self._dp

    @dp.setter
    def dp(self, value):
        self._assert_2X_ndarray('dp', value)
        self._dp = value

    @property
    def ddp(self):
        return self._ddp

    @ddp.setter
    def ddp(self, value):
        self._assert_2X_ndarray('ddp', value)
        self._ddp = value

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, value):
        self._assert_1X_ndarray('theta', value)
        self._theta = value

    @property
    def heading(self):
        return self._heading

    @heading.setter
    def heading(self, value):
        self._assert_2X_ndarray('heading', value)
        self._heading = value

    @property
    def d_b2b_center(self):
        return self._d_b2b_center

    @d_b2b_center.setter
    def d_b2b_center(self, value):
        self._d_b2b_center = value

    @property
    def is_collide_b2b(self):
        return self._is_collide_b2b

    @is_collide_b2b.setter
    def is_collide_b2b(self, value):
        self._is_collide_b2b = value

    @property
    def energy(self):
        return self._energy

    @energy.setter
    def energy(self, value):
        self._energy = value


    ## Environment Parameters

    @property
    def n_p(self):
        return self._n_p

    @n_p.setter
    def n_p(self, value:int):
        self._n_p = value

    @property
    def n_e(self):
        return self._n_e

    @n_e.setter
    def n_e(self, value:int):
        self._n_e = value
    
    @property
    def is_periodic(self):
        return self._is_periodic

    @is_periodic.setter
    def is_periodic(self, new_is_periodic):
        self._is_periodic = new_is_periodic

    @property
    def pursuer_strategy(self):
        return self._pursuer_strategy

    @pursuer_strategy.setter
    def pursuer_strategy(self, value:str):
        domain = ['input', 'static', 'random', 'nearest']
        if value not in domain:
            raise ValueError(f"reward_sharing_mode must be '{domain}'.")
        self._pursuer_strategy = value
    
    @property
    def escaper_strategy(self):
        return self._escaper_strategy

    @escaper_strategy.setter
    def escaper_strategy(self, value:str):
        domain = ['input', 'static', 'random', 'nearest']
        if value not in domain:
            raise ValueError(f"reward_sharing_mode must be '{domain}'.")
        self._escaper_strategy = value

    @property
    def billiards_mode(self):
        return self._billiards_mode

    @billiards_mode.setter
    def billiards_mode(self, value:bool):
        self._billiards_mode = value
        if value:
            self._dynamics_mode = 'Cartesian'
            self._is_periodic = False
    
    @property
    def reward_sharing_mode(self):
        return self._reward_sharing_mode

    @reward_sharing_mode.setter
    def reward_sharing_mode(self, new_reward_sharing_mode:str):
        if new_reward_sharing_mode not in ['sharing_mean', 'sharing_max', 'individual']:
            raise ValueError("reward_sharing_mode must be ['sharing_mean', 'sharing_max', 'individual'].")
        self._reward_sharing_mode = new_reward_sharing_mode

    @property
    def penalize_control_effort(self):
        return self._penalize_control_effort

    @penalize_control_effort.setter
    def penalize_control_effort(self, value):
        self._penalize_control_effort = value

    @property
    def penalize_collide_walls(self):
        return self._penalize_collide_walls

    @penalize_collide_walls.setter
    def penalize_collide_walls(self, value):
        self._penalize_collide_walls = value

    @property
    def penalize_distance(self):
        return self._penalize_distance

    @penalize_distance.setter
    def penalize_distance(self, value):
        self._penalize_distance = value

    @property
    def penalize_collide_agents(self):
        return self._penalize_collide_agents

    @penalize_collide_agents.setter
    def penalize_collide_agents(self, value):
        self._penalize_collide_agents = value

    @property
    def penalize_collide_obstacles(self):
        return self._penalize_collide_obstacles

    @penalize_collide_obstacles.setter
    def penalize_collide_obstacles(self, value):
        self._penalize_collide_obstacles = value

    
    @property
    def FoV_p(self):
        return self._FoV_p

    @FoV_p.setter
    def FoV_p(self, value):
        self._FoV_p = value

    @property
    def FoV_e(self):
        return self._FoV_e

    @FoV_e.setter
    def FoV_e(self, value):
        self._FoV_e = value

    @property
    def topo_n_p2e(self):
        return self._topo_n_p2e

    @topo_n_p2e.setter
    def topo_n_p2e(self, value):
        self._assert_nonnegative_int('topo_n_p2e', value)
        self._topo_n_p2e = value

    @property
    def topo_n_e2p(self):
        return self._topo_n_e2p

    @topo_n_e2p.setter
    def topo_n_e2p(self, value):
        self._assert_nonnegative_int('topo_n_e2p', value)
        self._topo_n_e2p = value

    @property
    def topo_n_p2p(self):
        return self._topo_n_p2p

    @topo_n_p2p.setter
    def topo_n_p2p(self, value):
        self._assert_nonnegative_int('topo_n_p2p', value)
        self._topo_n_p2p = value

    @property
    def topo_n_e2e(self):
        return self._topo_n_e2e

    @topo_n_e2e.setter
    def topo_n_e2e(self, value):
        self._assert_nonnegative_int('topo_n_e2e', value)
        self._topo_n_e2e = value


    @property
    def m_p(self):
        return self._m_p

    @m_p.setter
    def m_p(self, new_m_p):
        self._m_p = new_m_p
    
    @property
    def m_e(self):
        return self._m_e

    @m_e.setter
    def m_e(self, new_m_e):
        self._m_e = new_m_e

    
    @property
    def size_p(self):
        return self._size_p

    @size_p.setter
    def size_p(self, value):
        self._size_p = value

    @property
    def size_e(self):
        return self._size_e

    @size_e.setter
    def size_e(self, value):
        self._size_e = value

    @property
    def size_o(self):
        return self._size_o

    @size_o.setter
    def size_o(self, value):
        self._size_o = value
    

    @property
    def dynamics_mode(self):
        return self._dynamics_mode
    
    @dynamics_mode.setter
    def dynamics_mode(self, mode:str):
        if mode not in ['Cartesian', 'Polar']:
            raise ValueError("dynamics_mode must be 'Cartesian' or 'Polar', check your arguments.")
        self._dynamics_mode = mode
    
    
    @property
    def linVel_p_max(self):
        return self._linVel_p_max

    @linVel_p_max.setter
    def linVel_p_max(self, value):
        self._linVel_p_max = value

    @property
    def linVel_e_max(self):
        return self._linVel_e_max

    @linVel_e_max.setter
    def linVel_e_max(self, value):
        self._linVel_e_max = value

    @property
    def linAcc_p_max(self):
        return self._linAcc_p_max

    @linAcc_p_max.setter
    def linAcc_p_max(self, value):
        self._linAcc_p_max = value

    @property
    def linAcc_e_max(self):
        return self._linAcc_e_max

    @linAcc_e_max.setter
    def linAcc_e_max(self, value):
        self._linAcc_e_max = value
    
    @property
    def angle_p_max(self):
        return self._angle_p_max

    @angle_p_max.setter
    def angle_p_max(self, value):
        self._angle_p_max = value
    
    @property
    def angle_e_max(self):
        return self._angle_e_max

    @angle_e_max.setter
    def angle_e_max(self, value):
        self._angle_e_max = value


    @property
    def L(self):
        return self._L

    @L.setter
    def L(self, value):
        self._L = value

    @property
    def k_ball(self):
        return self._k_ball

    @k_ball.setter
    def k_ball(self, value):
        self._k_ball = value

    @property
    def k_wall(self):
        return self._k_wall

    @k_wall.setter
    def k_wall(self, value):
        self._k_wall = value

    @property
    def c_wall(self):
        return self._c_wall

    @c_wall.setter
    def c_wall(self, value):
        self._c_wall = value

    @property
    def c_aero(self):
        return self._c_aero

    @c_aero.setter
    def c_aero(self, value):
        self._c_aero = value
    
    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, value):
        if value > 0.5:
            print("Note: Please exercise caution as the chosen time step may potentially lead to unstable behaviors.")
        self._dt = value
    

    @property
    def render_traj(self):
        return self._render_traj

    @render_traj.setter
    def render_traj(self, value:bool):
        self._render_traj = value

    @property
    def traj_len(self):
        return self._traj_len

    @traj_len.setter
    def traj_len(self, value):
        self._assert_nonnegative_int('traj_len', value)
        self._traj_len = value

    @property
    def save_frame(self):
        return self._save_frame

    @save_frame.setter
    def save_frame(self, value:bool):
        self._save_frame = value


    @classmethod
    def _assert_nonnegative_int(cls, name, value):
        if not isinstance(value, int) or value < 0:
            raise TypeError(f" '{name}' must be a non-negative integer ")

    def _assert_2X_ndarray(cls, name, value):
        if not isinstance(value, np.ndarray) or value.shape[0] != 2:
            raise TypeError(f" '{name}' must be a 2-D np.ndarray with shape (2, x)")

    def _assert_1X_ndarray(cls, name, value):
        if not isinstance(value, np.ndarray) or value.shape[0] != 1:
            raise TypeError(f" '{name}' must be a 2-D np.ndarray with shape (1, x)")