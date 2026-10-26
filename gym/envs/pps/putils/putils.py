import numpy as np


def get_dist_b2b_batch(positions, half_width, periodic, pair_sizes):
    """批量计算粒子间距离，保留原生函数的周期边界和接触判定规则。"""
    # 与原生实现使用相同精度，避免接触阈值附近因类型转换改变碰撞结果。
    positions = np.asarray(positions, dtype=np.float64)
    delta = positions[:, :, None] - positions[:, None, :]
    if periodic:
        delta = make_periodic(delta, half_width)
    center = np.sqrt(delta[0] ** 2 + delta[1] ** 2)
    edge = np.abs(center - pair_sizes)
    # 恰好接触不算碰撞，保持原实现的严格小于判断。
    colliding = center < pair_sizes
    return center, edge, colliding


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


def get_focused_batch(positions, velocities, norm_threshold, width, remove_self):
    """输入轴为 (观察者, 候选邻居, xy)，输出固定 width 个邻居，不足时补零。"""
    shape = (positions.shape[0], width, 2)
    target_positions = np.zeros(shape)
    target_velocities = np.zeros(shape)
    norms = np.sqrt(positions[:, :, 0] ** 2 + positions[:, :, 1] ** 2)
    order = np.argsort(norms, axis=1)
    # 同种邻居沿用 get_focused 的规则：排序后丢弃第一项，而非按个体编号排除。
    # 即使多个距离并列为零，也必须保留这一规则，才能兼容原观测结果。
    offset = int(remove_self)
    selected = order[:, offset:offset + width]
    count = selected.shape[1]
    if count:
        # 距离严格小于视野阈值才可见；恰好位于边界的个体也按原规则补零。
        visible = np.take_along_axis(norms, selected, axis=1) < norm_threshold
        # 增加坐标轴以便同时提取 xy 两个分量，位置与速度使用相同邻居索引。
        selected = selected[:, :, None]
        target_positions[:, :count] = np.where(
            visible[:, :, None], np.take_along_axis(positions, selected, axis=1), 0.0,
        )
        target_velocities[:, :count] = np.where(
            visible[:, :, None], np.take_along_axis(velocities, selected, axis=1), 0.0,
        )
    return target_positions, target_velocities
