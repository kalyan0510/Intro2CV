import numpy as np
import cv2 as cv
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import multiprocess as mp


def get_pixel_derivatives(im_t0, im_t1):
    t0, t1 = im_t0.astype(np.float32), im_t1.astype(np.float32)
    ix, iy = np.zeros(t0.shape, dtype=np.float32), np.zeros(t0.shape, dtype=np.float32)
    ix[1:-1, 1:-1, ...] = t0[1:-1, 2:, ...] - t0[1:-1, :-2, ...]
    iy[1:-1, 1:-1, ...] = t0[2:, 1:-1, ...] - t0[:-2, 1:-1, ...]
    return ix, iy, (t0 - t1).astype(np.float32)


def lucas_kanade(im0, im1, k_shape=None):
    if k_shape is None:
        win_sz = min(max(3, min(im0.shape[:2]) / 4), 50)
        win_sz = int(2 * (win_sz // 2) + 1)
        k_shape = (win_sz, win_sz)
    print(k_shape)
    ix, iy, it = get_pixel_derivatives(im0, im1)
    grad_prods = [ix * ix, ix * iy, iy * iy, ix * it, iy * it]
    grad_prods = [np.sum(grad_prod, axis=2) for grad_prod in grad_prods] if len(im0.shape) == 3 else grad_prods
    wgps = [cv.GaussianBlur(grad_prod, k_shape, 0) for grad_prod in grad_prods]
    wgps = np.concatenate([wgp[:, :, np.newaxis] for wgp in wgps], axis=2)
    m_flow = np.zeros(im0.shape[:2] + (2,), dtype=np.float32)
    (w_h, w_l) = map(lambda x: (x - 1) // 2, k_shape)

    def compute_uv(i, j):
        a = [[wgps[i, j, 0], wgps[i, j, 1]], [wgps[i, j, 1], wgps[i, j, 2]]]
        b = [[-wgps[i, j, 3]], [-wgps[i, j, 4]]]
        uv, _, rank, _ = np.linalg.lstsq(a, b, rcond=0.2)
        return uv.flatten() if rank == 2 else [0, 0]

    with mp.Pool(10) as pool:
        def i_slice(i_var):
            uv_slice = np.zeros((im0.shape[1], 2))
            for j in range(w_l, im0.shape[1] - w_l):
                u, v = compute_uv(i_var, j)
                uv_slice[j, :] = [u, v]
            return uv_slice

        i_range = range(w_h, im0.shape[0] - w_h)
        slices = pool.map(i_slice, list(i_range))
        for i, slice_i in zip(i_range, slices):
            m_flow[i, :, :] = slice_i
    return -m_flow


def get_flow_arrows(flow, gap=None, show_arrows_on_plt=False):
    gap = [gap, flow.shape[0] // 30][gap is None]
    gap = max(gap, 1)
    x = np.arange(0, flow.shape[1], 1)
    y = np.arange(0, flow.shape[0], 1)
    x, y = np.meshgrid(x, y)
    fig = plt.figure()
    axs = fig.add_axes([0, 0, 1, 1], frameon=False)
    fig.patch.set_alpha(0)
    axs.patch.set_alpha(0)
    canvas = FigureCanvas(fig)
    plt.quiver(x[::gap, ::gap], y[::-gap, ::-gap], flow[::gap, ::gap, 0], -flow[::gap, ::gap, 1], color='red')
    axs.axis('off')
    axs.margins(0)
    canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    if not show_arrows_on_plt:
        plt.close(fig)
    return cv.resize(image_from_plot, dsize=flow.shape[1::-1])


def add_flow_over_im(im, flow, gap=None, show_arrows_on_plt=False):
    im = im.copy()
    if len(im.shape) == 2:
        im = cv.cvtColor(im, cv.COLOR_GRAY2RGB)
    arrows_arbg = get_flow_arrows(flow, gap=gap, show_arrows_on_plt=show_arrows_on_plt)
    alpha = arrows_arbg[:, :, 0]
    for i in range(3):
        im[:, :, i] = (alpha / 255.0) * arrows_arbg[:, :, 1 + i] + ((255.0 - alpha) / 255.0) * im[:, :, i]
    return im


def reduce(im):
    im_blur = cv.GaussianBlur(im, (5, 5), 0)
    return im_blur[::2, ::2, ...]


def expand(im, dst_size, interpolation=None):
    return cv.GaussianBlur(cv.resize(im, dst_size[1::-1], interpolation=interpolation or cv.INTER_LINEAR), (5, 5), 0)


def gaussian_pyramid(im, max_l=np.PINF, up_scaled=False, use_cv=False):
    levels = min(int(np.log2(min(im.shape[:2])) + 1), max_l)
    g_pyr = [im]
    if use_cv:
        subscale = lambda i: cv.pyrUp(cv.pyrDown(i), i.shape) if up_scaled else cv.pyrDown(i)
    else:
        subscale = lambda i: cv.pyrUp(cv.pyrDown(i), i.shape) if up_scaled else cv.pyrDown(i)
    [g_pyr.append(subscale(g_pyr[-1])) for i in range(levels - 1)]
    return g_pyr


def laplacian_pyramid(im, max_l=np.PINF, use_cv=False):
    levels = min(int(np.log2(min(im.shape[:2])) + 1), max_l)
    l_pyr = [im]
    for i in range(levels - 1):
        tmp = reduce(l_pyr[-1]) if not use_cv else cv.pyrDown(l_pyr[-1])
        expand_gauss = (expand if not use_cv else cv.pyrUp)(tmp, l_pyr[-1].shape).astype(np.int32)
        l_pyr[-1] = l_pyr[-1].astype(np.int32) - expand_gauss
        l_pyr.append(tmp)
    return l_pyr


def remap(a, flow):
    h, w = flow.shape[:2]
    flow_map = -flow.copy()
    flow_map[:, :, 0] += np.arange(w)
    flow_map[:, :, 1] += np.arange(h)[:, np.newaxis]
    warped = cv.remap(a, flow_map.astype(np.float32), None, cv.INTER_LINEAR)
    return warped


def hierarchical_lk(im1, im2, k_shape=None, max_l=np.PINF, up_scaled=False):
    gpy_1, gpy_2 = gaussian_pyramid(im1, up_scaled=up_scaled), gaussian_pyramid(im2, up_scaled=up_scaled)
    max_l = min(len(gpy_1), len(gpy_2), max_l)
    flow = np.zeros(gpy_1[-1].shape[:2] + (2,)).astype(np.float32)
    for i in range(max_l - 1, -1, -1):
        flow = 2.0 * expand(flow, gpy_2[i].shape[:2] + (2,), interpolation=cv.INTER_LINEAR)
        warped = remap(gpy_1[i], flow)
        flow += lucas_kanade(warped, gpy_2[i], k_shape)
    return flow


def hierarchical_laplacian_lk(im1, im2, k_shape=None, max_l=np.PINF):
    gpy_1, gpy_2 = laplacian_pyramid(im1[...,0] + cv.Canny(im1, 100,200)), laplacian_pyramid(im2[...,0] + cv.Canny(im2, 100,200))
    gpy_1, gpy_2 = [((level+255)/2).astype(np.uint8) for level in gpy_1], [((level+255)/2).astype(np.uint8) for level in gpy_2]
    max_l = min(len(gpy_1), len(gpy_2), max_l)
    flow = np.zeros(gpy_1[-1].shape[:2] + (2,)).astype(np.float32)
    for i in range(max_l - 1, -1, -1):
        flow = 2.0 * expand(flow, gpy_2[i].shape[:2] + (2,), interpolation=cv.INTER_LINEAR)
        warped = remap(gpy_1[i], flow)
        flow += lucas_kanade(warped, gpy_2[i], k_shape)
    return flow
