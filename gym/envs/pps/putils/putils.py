import numpy as np


def make_periodic(x:np.array, L:float) -> np.array:
    x[x > L] -= 2 * L 
    x[x < -L] += 2 * L
    return x


def normalize_angle(x:np.array) -> np.array:
    return ((x + np.pi) % (2 * np.pi)) - np.pi


def get_sizes(size_p, size_e, size_o, n_p, n_e, n_o):
    n_peo = n_p + n_e + n_o 
    size = np.concatenate((
        np.full(n_p, size_p),
        np.full(n_e, size_e),
        np.full(n_o, size_o)
    ))
    sizes = np.tile(size.reshape(n_peo, 1), (1, n_peo))
    sizes = sizes + sizes.T
    np.fill_diagonal(sizes, 0)
    return size, sizes


def get_mass(m_p, m_e, m_o, n_p, n_e, n_o):
    masses = np.concatenate((
        np.full(n_p, m_p),
        np.full(n_e, m_e),
        np.full(n_o, m_o)
    ))
    return masses


def get_focused(Pos, Vel, norm_threshold, width, remove_self):
    norms = np.sqrt( Pos[0,:]**2 + Pos[1,:]**2 )
    sorted_seq = np.argsort(norms)    
    Pos = Pos[:, sorted_seq]   
    norms = norms[sorted_seq] 
    Pos = Pos[:, norms < norm_threshold] 
    sorted_seq = sorted_seq[norms < norm_threshold]   
    if remove_self == True:
        Pos = Pos[:,1:]  
        sorted_seq = sorted_seq[1:]                    
    Vel = Vel[:, sorted_seq]
    target_Pos = np.zeros( (2, width) )
    target_Vel = np.zeros( (2, width) )
    until_idx = np.min( [Pos.shape[1], width] )
    target_Pos[:, :until_idx] = Pos[:, :until_idx] 
    target_Vel[:, :until_idx] = Vel[:, :until_idx]
    return target_Pos, target_Vel   
