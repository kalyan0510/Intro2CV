import numpy as np


def moment_p_q(im, p, q, x_=.0, y_=.0):
    x_ = int(x_)
    y_ = int(y_)
    cx = (np.arange(im.shape[1]) - x_)
    cy = (np.arange(im.shape[0]) - y_).reshape((-1, 1))
    return np.sum(cx ** p * cy ** q * im)


def scale_inv_m(im, p, q, x_=.0, y_=.0, m_00=None):
    m_00 = m_00 if m_00 is not None else moment_p_q(im, 0, 0)
    return moment_p_q(im, p, q, x_, y_) / m_00 ** (1 + (p + q) / 2)


def compute_moment_vector(motion_im):
    m_vec = [(2, 0), (0, 2), (1, 2), (2, 1), (2, 2), (3, 0), (0, 3)]
    m_00 = moment_p_q(motion_im, 0, 0)
    x_ = moment_p_q(motion_im, 1, 0) / m_00
    y_ = moment_p_q(motion_im, 0, 1) / m_00
    return [scale_inv_m(motion_im, p, q, x_, y_, m_00) for p, q in m_vec]


def compute_mhi_feature_vec(motion_im, with_mei=True):
    if with_mei:
        return np.concatenate([compute_moment_vector(motion_im),
                               compute_moment_vector((motion_im > 0).astype(np.float32) * motion_im.max())], axis=0)
    else:
        return np.asarray(compute_moment_vector(motion_im))
