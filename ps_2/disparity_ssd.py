from ps_hepers.helpers import get_window_ix
import numpy as np
import multiprocess as mp
from multiprocessing import freeze_support


def disparity_ssd(left, right, max_disp=None, window_shape=None):
    """
    Compute disparity map D(y, x) such that: L(y, x) = R(y, x + D(y, x))
    :param left: Grayscale left image
    :param right: Grayscale right image, same size as L
    :param max_disp: The max disparity. This constraint helps in faster disparity checking
    :param window_shape: shape of kenel
    :return: Disparity map, same size as L, R
    """
    global compute_stereo_on_slice
    freeze_support()
    max_disp = [max_disp, left.shape[1] // 6][max_disp is None]
    window_shape = [window_shape, (3, 3)][window_shape is None]
    disparity_map = np.zeros(left.shape)
    (w_h, w_l) = map(lambda x: (x - 1) // 2, window_shape)

    def compute_stereo_on_slice(i):
        print(i)
        d_map_slice = np.zeros((left.shape[1],))
        for j in range(w_l, left.shape[1] - w_l):
            (d, ssd) = (0, np.PINF)
            for j_i in range(max(w_l, j - max_disp), min(left.shape[1] - w_l, j + max_disp)):
                l_w = left[get_window_ix(left.shape, (i, j), window_shape)] * 1.0
                r_w = right[get_window_ix(right.shape, (i, j_i), window_shape)] * 1.0
                ssd_i = np.sum(np.square(np.subtract(l_w, r_w)))
                (d, ssd) = (j_i - j if ssd_i < ssd else d, min(ssd_i, ssd))
            d_map_slice[j] = d
        return d_map_slice

    with mp.Pool(6) as pool:
        slices = pool.map(compute_stereo_on_slice, list(range(w_h, left.shape[0] - w_h)))
        for i, slice_i in zip(range(w_h, left.shape[0] - w_h), slices):
            disparity_map[i, :] = slice_i
    return disparity_map
