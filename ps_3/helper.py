import numpy as np


def construct_a_mat(pts_world, pts_proj):
    a = []
    for (x, y, z), (u, v) in zip(pts_world, pts_proj):
        a.append([x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u])
        a.append([0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z, -v])
    return np.asarray(a)


def construct_f_mat(pts_im_left, pts_im_right):
    f = []
    for (u, v, *rest), (u_, v_, *rest) in zip(pts_im_left, pts_im_right):
        f.append([u_ * u, u_ * v, u_, v_ * u, v_ * v, v_, u, v, 1])
    return np.asarray(f)


def solve_for_svd(a):
    w, v = np.linalg.eig(np.matmul(a.T, a))
    m = v[:, w.argmin()]
    return m


def solve_with_leastsq(a):
    b = -a[:, -1]
    a = a[:, 0:-1]
    m, res, _, _ = np.linalg.lstsq(a, b, rcond=None)
    return np.append(m, 1)


def solve_for_m_svd(pts_world, pts_proj):
    a = construct_a_mat(pts_world, pts_proj)
    return solve_for_svd(a).reshape((3, 4))


def solve_for_m_leastsq(pts_world, pts_proj):
    a = construct_a_mat(pts_world, pts_proj)
    return solve_with_leastsq(a).reshape((3, 4))


def solve_for_f_svd(pts_im1, pts_im2):
    a = construct_f_mat(pts_im1, pts_im2)
    return solve_for_svd(a).reshape((3, 3))


def solve_for_f_leastsq(pts_im1, pts_im2):
    a = construct_f_mat(pts_im1, pts_im2)
    return solve_with_leastsq(a).reshape((3, 3))


def fix_f_rank(f):
    u, s, vh = np.linalg.svd(f)
    s[s.argmin()] = 0
    return np.matmul(np.matmul(u, np.diag(s)), vh)


def get_epipolar_line_ends(f, pts_b, im_shape, t=None):
    pts_b_t = np.append(np.asarray(pts_b), np.ones((len(pts_b), 1)), axis=1).T
    l_l = np.cross([0, 0, 1], [im_shape[0], 0, 1])
    l_r = np.cross([0, im_shape[1], 1], [im_shape[0], im_shape[1], 1])
    l_a = np.matmul(f.T, pts_b_t)
    if t is not None:
        l_a = np.matmul(t.T, l_a)
    skew = lambda x: np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
    p_l = np.matmul(skew(l_l).T, l_a).T
    p_r = np.matmul(skew(l_r).T, l_a).T
    p_l = np.asarray([(int(x[0] / x[2]), int(x[1] / x[2])) for x in p_l])
    p_r = np.asarray([(int(x[0] / x[2]), int(x[1] / x[2])) for x in p_r])
    return tuple(zip(p_l, p_r))


def get_transformation_mat_2d(points):
    s_a = 1 / np.max(points)
    t_a_s = np.diag([s_a, s_a, 1])
    c_a = np.mean(points, axis=0)
    t_a_c = np.asarray([[1, 0, -c_a[0]], [0, 1, -c_a[1]], [0, 0, 1]])
    return np.matmul(t_a_s, t_a_c)


def calculate_avg_residual(m, pts_world, pts_proj):
    pts_world_mat = np.append(np.asarray(pts_world), np.ones((len(pts_world), 1)), axis=1).T
    pts_proj_est = np.matmul(m, pts_world_mat)
    pts_proj_est_non_homo = pts_proj_est.T[:, 0:2] / pts_proj_est.T[:, [2, 2]]
    residual = np.linalg.norm(pts_proj_est_non_homo - np.asarray(pts_proj))
    # return averaged residual
    return residual/np.sqrt(pts_world_mat.shape[0])
